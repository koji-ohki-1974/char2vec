package main

import (
	"bufio"
	"compress/bzip2"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const MAX_UINT = ^uint(0)
const MAX_INT = int(MAX_UINT >> 1)

const EXP_TABLE_SIZE int = 1000
const MAX_EXP float64 = 6.
const MAX_SENTENCE_LENGTH int = 1000
const MAX_CODE_LENGTH int = 40

const SEEK_SET int = 0

const vocab_hash_size int = 30000000 // Maximum 30 * 0.7 = 21M characters in the vocabulary

//type real float64 // Precision of float numbers

type vocab_char struct {
	cn      int64
	point   []int
	char    rune
	code    []byte
	codelen byte
}

type vocab_slice []vocab_char

func (me vocab_slice) Len() int {
	return len(me)
}

func (me vocab_slice) Less(i, j int) bool {
	return me[i].cn > me[j].cn
}

func (me vocab_slice) Swap(i, j int) {
	tmp := me[i]
	me[i] = me[j]
	me[j] = tmp
}

var train_file, output_file string
var save_vocab_file, read_vocab_file string
var vocab vocab_slice
var binaryf int = 0
var cbow int = 1
var debug_mode int = 2
var window int = 5
var min_count int64 = 5
var num_threads int = 12
var min_reduce int64 = 1
var vocab_hash map[rune]int = map[rune]int{}
var vocab_max_size int = 1000
var vocab_size int = 0
var layer1_size int = 100
var train_chars int64 = 0
var char_count_actual int64 = 0
var iter int = 5
var file_size int64 = 0
var classes int = 0
var alpha float64 = 0.025
var starting_alpha float64
var sample float64 = 1e-3
var syn0 []float64
var syn1 []float64
var syn1neg []float64
var expTable []float64
var start time.Time

var hs int = 0
var negative int = 5

const table_size int = 1e8

var table []int

var m *sync.Mutex = new(sync.Mutex)

func InitUnigramTable() {
	fmt.Fprintln(os.Stderr, "InitUnigramTable")
	var train_chars_pow float64 = 0
	var d1 float64
	var power float64 = 0.75
	table = make([]int, table_size)
	for a := 0; a < vocab_size; a++ {
		train_chars_pow += math.Pow(float64(vocab[a].cn), power)
	}
	i := 0
	d1 = math.Pow(float64(vocab[i].cn), power) / train_chars_pow
	for a := 0; a < table_size; a++ {
		table[a] = i
		if float64(a)/float64(table_size) > d1 {
			i++
			d1 += math.Pow(float64(vocab[i].cn), power) / train_chars_pow
		}
		if i >= vocab_size {
			i = vocab_size - 1
		}
	}
}

// Returns position of a character in the vocabulary; if the character is not found, returns -1
func SearchVocab(char rune) int {
	i, ok := vocab_hash[char]
	if !ok {
		return -1
	}
	return i
}

// Reads a character and returns its index in the vocabulary
func ReadCharIndex(fin *bufio.Reader) (int, error) {
	var char rune
	char, _, err := fin.ReadRune()
	if err == io.EOF {
		return -1, err
	}
	return SearchVocab(char), nil
}

// Adds a character to the vocabulary
func AddCharToVocab(char rune) int {
	vocab[vocab_size].char = char
	vocab[vocab_size].cn = 0
	vocab_size++
	// Reallocate memory if needed
	if vocab_size+2 >= vocab_max_size {
		vocab_max_size += 1000
		vocab = append(vocab, make([]vocab_char, 1000)...)
	}
	vocab_hash[char] = vocab_size - 1
	return vocab_size - 1
}

// Sorts the vocabulary by frequency using character counts
func SortVocab() {
	fmt.Fprintln(os.Stderr, "SortVocab")
	// Sort the vocabulary and keep </s> at the first position
	sort.Sort(vocab[1:])
	vocab_hash = map[rune]int{}
	size := vocab_size
	train_chars = 0
	for a := 0; a < size; a++ {
		// Characters occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) && (a != 0) {
			vocab_size--
			vocab[a].char = 0
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			vocab_hash[vocab[a].char] = a
			train_chars += int64(vocab[a].cn)
		}
	}
	vocab = vocab[:vocab_size+1]
	// Allocate memory for the binary tree construction
	for a := 0; a < vocab_size; a++ {
		vocab[a].code = make([]byte, MAX_CODE_LENGTH)
		vocab[a].point = make([]int, MAX_CODE_LENGTH)
	}
}

// Reduces the vocabulary by removing infrequent tokens
func ReduceVocab() {
	fmt.Fprintln(os.Stderr, "ReduceVocab")
	var b int = 0
	for a := 0; a < vocab_size; a++ {
		if vocab[a].cn > min_reduce {
			vocab[b].cn = vocab[a].cn
			vocab[b].char = vocab[a].char
			b++
		} else {
			vocab[a].char = 0
		}
	}
	vocab_size = b
	vocab_hash = map[rune]int{}
	for a := 0; a < vocab_size; a++ {
		// Hash will be re-computed, as it is not actual
		vocab_hash[vocab[a].char] = a
	}
	//  fflush(stdout);
	min_reduce++
}

// Create binary Huffman tree using the character counts
// Frequent characters will have short uniqe binary codes
func CreateBinaryTree() {
	fmt.Fprintln(os.Stderr, "CreateBinaryTree")
	var min1i, min2i, pos1, pos2 int
	var point []int = make([]int, MAX_CODE_LENGTH)
	var code []byte = make([]byte, MAX_CODE_LENGTH)
	var count []int64 = make([]int64, vocab_size*2+1)
	var binaryt []int = make([]int, vocab_size*2+1)
	var parent_node []int = make([]int, vocab_size*2+1)
	for a := 0; a < vocab_size; a++ {
		count[a] = int64(vocab[a].cn)
	}
	for a := vocab_size; a < vocab_size*2; a++ {
		count[a] = 1e15
	}
	pos1 = vocab_size - 1
	pos2 = vocab_size
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for a := 0; a < vocab_size-1; a++ {
		// First, find two smallest nodes 'min1, min2'
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min1i = pos1
				pos1--
			} else {
				min1i = pos2
				pos2++
			}
		} else {
			min1i = pos2
			pos2++
		}
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min2i = pos1
				pos1--
			} else {
				min2i = pos2
				pos2++
			}
		} else {
			min2i = pos2
			pos2++
		}
		count[vocab_size+a] = count[min1i] + count[min2i]
		parent_node[min1i] = vocab_size + a
		parent_node[min2i] = vocab_size + a
		binaryt[min2i] = 1
	}
	// Now assign binary code to each vocabulary character
	for a := 0; a < vocab_size; a++ {
		b := a
		i := 0
		for {
			code[i] = byte(binaryt[b])
			point[i] = b
			i++
			b = parent_node[b]
			if b == vocab_size*2-2 {
				break
			}
		}
		vocab[a].codelen = byte(i)
		vocab[a].point[0] = vocab_size - 2
		for b = 0; b < i; b++ {
			vocab[a].code[i-b-1] = code[b]
			vocab[a].point[i-b] = point[b] - vocab_size
		}
	}
}

func LearnVocabFromTrainFile() {
	fmt.Fprintln(os.Stderr, "LearnVocabFromTrainFile")
	var char rune
	var fin *bufio.Reader
	var i int
	vocab_hash = map[rune]int{}
	f, err := os.Open(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: training data file not found!\n")
		os.Exit(1)
	}
	defer f.Close()
	if strings.HasSuffix(strings.ToLower(train_file), ".bz2") {
		fin = bufio.NewReader(bzip2.NewReader(f))
	} else {
		fin = bufio.NewReader(f)
	}
	vocab_size = 0
	AddCharToVocab(0)
	for {
		char, _, err = fin.ReadRune()
		if err == io.EOF {
			break
		}
		train_chars++
		if (debug_mode > 1) && (train_chars%1000000 == 0) {
			fmt.Fprintf(os.Stderr, "%dK%c", train_chars/1000, 13)
			//      fflush(stdout);
		}
		i = SearchVocab(char)
		if i == -1 {
			a := AddCharToVocab(char)
			vocab[a].cn = 1
		} else {
			vocab[i].cn++
		}
		if float64(vocab_size) > float64(vocab_hash_size)*0.7 {
			ReduceVocab()
		}
	}
	SortVocab()
	if debug_mode > 0 {
		fmt.Fprintf(os.Stderr, "Vocab size: %d\n", vocab_size)
		fmt.Fprintf(os.Stderr, "Characters in train file: %d\n", train_chars)
	}
	fi, _ := os.Stat(train_file)
	file_size = fi.Size()
}

func SaveVocab() {
	fmt.Fprintln(os.Stderr, "SaveVocab")
	f, _ := os.Create(save_vocab_file)
	defer f.Close()
	fo := bufio.NewWriter(f)
	for i := 0; i < vocab_size; i++ {
		fmt.Fprintf(fo, "%c %d\n", vocab[i].char, vocab[i].cn)
	}
	fo.Flush()
}

func ReadVocab() {
	fmt.Fprintln(os.Stderr, "ReadVocab")
	var i int64 = 0
	var c byte
	var char rune
	f, err := os.Open(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Vocabulary file not found\n")
		os.Exit(1)
	}
	defer f.Close()
	fin := bufio.NewReader(f)
	vocab_hash = map[rune]int{}
	vocab_size = 0
	for {
		char, _, err = fin.ReadRune()
		if err == io.EOF {
			break
		}
		a := AddCharToVocab(char)
		fmt.Fscanf(fin, "%d%c", &vocab[a].cn, &c)
		i++
	}
	SortVocab()
	if debug_mode > 0 {
		fmt.Fprintf(os.Stderr, "Vocab size: %d\n", vocab_size)
		fmt.Fprintf(os.Stderr, "Characters in train file: %d\n", train_chars)
	}
	fi, err := os.Stat(train_file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: training data file not found!\n")
		os.Exit(1)
	}
	file_size = fi.Size()
}

func InitNet() {
	fmt.Fprintln(os.Stderr, "InitNet")
	var next_random uint64 = 1
	syn0 = make([]float64, vocab_size*layer1_size)
	if hs != 0 {
		syn1 = make([]float64, vocab_size*layer1_size)
		for a := 0; a < vocab_size; a++ {
			for b := 0; b < layer1_size; b++ {
				syn1[a*layer1_size+b] = 0
			}
		}
	}
	if negative > 0 {
		syn1neg = make([]float64, vocab_size*layer1_size)
		for a := 0; a < vocab_size; a++ {
			for b := 0; b < layer1_size; b++ {
				syn1neg[a*layer1_size+b] = 0
			}
		}
	}
	for a := 0; a < vocab_size; a++ {
		for b := 0; b < layer1_size; b++ {
			next_random = next_random*uint64(25214903917) + 11
			syn0[a*layer1_size+b] = ((float64(next_random&0xFFFF) / float64(65536)) - 0.5) / float64(layer1_size)
		}
	}
	CreateBinaryTree()
}

func PrepareTrainFileReader(id int, f *os.File, bz2 bool) *bufio.Reader {
	var br *bufio.Reader
	if bz2 {
		br = bufio.NewReader(bzip2.NewReader(f))
		n := train_chars / int64(num_threads) * int64(id)
		bufsize := 4096
		buf := make([]byte, bufsize)
		for ; n > int64(bufsize); n -= int64(bufsize) {
			_, err := br.Read(buf)
			if err != nil {
				panic(err)
			}
		}
		buf = make([]byte, n)
		_, err := br.Read(buf)
		if err != nil {
			panic(err)
		}
	} else {
		f.Seek(file_size/int64(num_threads)*int64(id), SEEK_SET)
		br = bufio.NewReader(f)
	}
	return br
}

func TrainModelThread(id int) {
	fmt.Fprintln(os.Stderr, "TrainModelThread")
	var a, b, d, cw, char, last_char int
	var sentence_length, sentence_position int = 0, 0
	var char_count, last_char_count int64 = 0, 0
	var sen []int = make([]int, MAX_SENTENCE_LENGTH+1)
	var l1, l2, c, target, label int
	var local_iter int = iter
	var next_random uint64 = uint64(id)
	var f, g float64
	var now time.Time
	var neu1 []float64 = make([]float64, layer1_size)
	var neu1e []float64 = make([]float64, layer1_size)
	fi, _ := os.Open(train_file)
	defer fi.Close()
	bz2 := strings.HasSuffix(strings.ToLower(train_file), ".bz2")
	br := PrepareTrainFileReader(id, fi, bz2)
	for {
		if char_count-last_char_count > 10000 {
			//			char_count_actual += char_count - last_char_count
			atomic.AddInt64(&char_count_actual, char_count-last_char_count)
			last_char_count = char_count
			if debug_mode > 1 {
				now = time.Now()
				fmt.Fprintf(os.Stderr, "%cAlpha: %f  Progress: %.2f%%  Characters/thread/sec: %.2fk  ", 13, alpha,
					float64(char_count_actual)/float64(int64(iter)*train_chars+1)*100,
					float64(char_count_actual)/(float64(now.Unix()-start.Unix()+1)*1000))
				//				fflush(stdout)
			}
			alpha = starting_alpha * (1 - float64(char_count_actual)/float64(int64(iter)*train_chars+1))
			if alpha < starting_alpha*0.0001 {
				alpha = starting_alpha * 0.0001
			}
		}
		var err error
		if sentence_length == 0 {
			for {
				char, err = ReadCharIndex(br)
				if err == io.EOF {
					break
				}
				if char == -1 {
					continue
				}
				char_count++
				if char == 0 {
					break
				}
				// The subsampling randomly discards frequent characters while keeping the ranking same
				if sample > 0 {
					var ran float64 = math.Sqrt(float64(vocab[char].cn)/(sample*float64(train_chars))) + 1*(sample*float64(train_chars))/float64(vocab[char].cn)
					next_random = next_random*25214903917 + 11
					if ran < float64(next_random&0xFFFF)/65536 {
						continue
					}
				}
				sen[sentence_length] = char
				sentence_length++
				if int(sentence_length) >= MAX_SENTENCE_LENGTH {
					break
				}
			}
			sentence_position = 0
		}
		if err == io.EOF || (char_count > train_chars/int64(num_threads)) {
			char_count_actual += char_count - last_char_count
			local_iter--
			if local_iter == 0 {
				break
			}
			char_count = 0
			last_char_count = 0
			sentence_length = 0
			br = PrepareTrainFileReader(id, fi, bz2)
			continue
		}
		char = sen[sentence_position]
		if char == -1 {
			continue
		}
		for c = 0; c < layer1_size; c++ {
			neu1[c] = 0
		}
		for c = 0; c < layer1_size; c++ {
			neu1e[c] = 0
		}
		next_random = next_random*uint64(25214903917) + 11
		b = int(next_random % uint64(window))
		if cbow != 0 { //train the cbow architecture
			// in -> hidden
			cw = 0
			for a = b; a < window*2+1-b; a++ {
				if a != window {
					c = sentence_position - window + a
					if c < 0 {
						continue
					}
					if c >= sentence_length {
						continue
					}
					last_char = sen[c]
					if last_char == -1 {
						continue
					}
					for c = 0; c < layer1_size; c++ {
						neu1[c] += syn0[c+last_char*layer1_size]
					}
					cw++
				}
			}
			if cw != 0 {
				for c = 0; c < layer1_size; c++ {
					neu1[c] /= float64(cw)
				}
				if hs != 0 {
					for d = 0; d < int(vocab[char].codelen); d++ {
						f = 0
						l2 = vocab[char].point[d] * layer1_size
						// Propagate hidden -> output
						for c = 0; c < layer1_size; c++ {
							f += neu1[c] * syn1[c+l2]
						}
						if f <= -MAX_EXP {
							continue
						} else if f >= MAX_EXP {
							continue
						} else {
							f = expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]
						}
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - float64(vocab[char].code[d]) - f) * alpha
						// Propagate errors output -> hidden
						for c = 0; c < layer1_size; c++ {
							neu1e[c] += g * syn1[c+l2]
						}
						// Learn weights hidden -> output
						for c = 0; c < layer1_size; c++ {
							syn1[c+l2] += g * neu1[c]
						}
					}
				}
				// NEGATIVE SAMPLING
				if negative > 0 {
					for d = 0; d < negative+1; d++ {
						if d == 0 {
							target = char
							label = 1
						} else {
							next_random = next_random*uint64(25214903917) + 11
							target = table[(next_random>>16)%uint64(table_size)]
							if target == 0 {
								target = int(next_random%uint64(vocab_size-1)) + 1
							}
							if target == char {
								continue
							}
							label = 0
						}
						l2 = target * layer1_size
						f = 0
						for c = 0; c < layer1_size; c++ {
							f += neu1[c] * syn1neg[c+l2]
						}
						if f > MAX_EXP {
							g = float64(label-1) * alpha
						} else if f < -MAX_EXP {
							g = float64(label-0) * alpha
						} else {
							g = (float64(label) - expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]) * alpha
						}
						for c = 0; c < layer1_size; c++ {
							neu1e[c] += g * syn1neg[c+l2]
						}
						for c = 0; c < layer1_size; c++ {
							syn1neg[c+l2] += g * neu1[c]
						}
					}
				}
				// hidden -> in
				for a = b; a < window*2+1-b; a++ {
					if a != window {
						c = sentence_position - window + a
						if c < 0 {
							continue
						}
						if c >= sentence_length {
							continue
						}
						last_char = sen[c]
						if last_char == -1 {
							continue
						}
						for c = 0; c < layer1_size; c++ {
							syn0[c+last_char*layer1_size] += neu1e[c]
						}
					}
				}
			}
		} else { //train skip-gram
			for a = b; a < window*2+1-b; a++ {
				if a != window {
					c = sentence_position - window + a
					if c < 0 {
						continue
					}
					if c >= sentence_length {
						continue
					}
					last_char = sen[c]
					if last_char == -1 {
						continue
					}
					l1 = last_char * layer1_size
					for c = 0; c < layer1_size; c++ {
						neu1e[c] = 0
					}
					// HIERARCHICAL SOFTMAX
					if hs != 0 {
						for d = 0; d < int(vocab[char].codelen); d++ {
							f = 0
							l2 = vocab[char].point[d] * layer1_size
							// Propagate hidden -> output
							for c = 0; c < layer1_size; c++ {
								f += syn0[c+l1] * syn1[c+l2]
							}
							if f <= -MAX_EXP {
								continue
							} else if f >= MAX_EXP {
								continue
							} else {
								f = expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]
							}
							// 'g' is the gradient multiplied by the learning rate
							g = (1 - float64(vocab[char].code[d]) - f) * alpha
							// Propagate errors output -> hidden
							for c = 0; c < layer1_size; c++ {
								neu1e[c] += g * syn1[c+l2]
							}
							// Learn weights hidden -> output
							for c = 0; c < layer1_size; c++ {
								syn1[c+l2] += g * syn0[c+l1]
							}
						}
					}
					// NEGATIVE SAMPLING
					if negative > 0 {
						for d = 0; d < negative+1; d++ {
							if d == 0 {
								target = char
								label = 1
							} else {
								next_random = next_random*uint64(25214903917) + 11
								target = table[(next_random>>16)%uint64(table_size)]
								if target == 0 {
									target = int(next_random%uint64(vocab_size-1)) + 1
								}
								if target == char {
									continue
								}
								label = 0
							}
							l2 = target * layer1_size
							f = 0
							for c = 0; c < layer1_size; c++ {
								f += syn0[c+l1] * syn1neg[c+l2]
							}
							if f > MAX_EXP {
								g = float64(label-1) * alpha
							} else if f < -MAX_EXP {
								g = float64(label-0) * alpha
							} else {
								g = (float64(label) - expTable[(int)((f+MAX_EXP)*(float64(EXP_TABLE_SIZE)/MAX_EXP/2))]) * alpha
							}
							for c = 0; c < layer1_size; c++ {
								neu1e[c] += g * syn1neg[c+l2]
							}
							for c = 0; c < layer1_size; c++ {
								syn1neg[c+l2] += g * syn0[c+l1]
							}
						}
					}
					// Learn weights input -> hidden
					for c = 0; c < layer1_size; c++ {
						syn0[c+l1] += neu1e[c]
					}
				}
			}
		}
		sentence_position++
		if sentence_position >= sentence_length {
			sentence_length = 0
			continue
		}
	}
}

func TrainModel() {
	fmt.Fprintln(os.Stderr, "TrainModel")
	var fo *bufio.Writer
	fmt.Fprintf(os.Stderr, "Starting training using file %s\n", train_file)
	starting_alpha = alpha
	if read_vocab_file != "" {
		ReadVocab()
	} else {
		LearnVocabFromTrainFile()
	}
	if save_vocab_file != "" {
		SaveVocab()
	}
	if output_file == "" {
		return
	}
	InitNet()
	if negative > 0 {
		InitUnigramTable()
	}
	start = time.Now()
	ch := make(chan int, num_threads)
	for a := 0; a < num_threads; a++ {
		go func(a int) {
			TrainModelThread(a)
			ch <- 0
		}(a)
	}
	for a := 0; a < num_threads; a++ {
		<-ch
	}
	f, _ := os.Create(output_file)
	defer f.Close()
	fo = bufio.NewWriter(f)
	if classes == 0 {
		// Save the character vectors
		fmt.Fprintf(fo, "%d %d\n", vocab_size, layer1_size)
		for a := 0; a < vocab_size; a++ {
			fmt.Fprintf(fo, "%c ", vocab[a].char)
			if binaryf != 0 {
				binary.Write(fo, binary.LittleEndian, syn0[a*layer1_size:(a+1)*layer1_size])
			} else {
				for b := 0; b < layer1_size; b++ {
					fmt.Fprintf(fo, "%f ", syn0[a*layer1_size+b])
				}
			}
			fmt.Fprintf(fo, "\n")
		}
	} else {
		// Run K-means on the character vectors
		var clcn int = classes
		var iter int = 10
		var closeid int
		var centcn []int = make([]int, classes)
		var cl []int = make([]int, vocab_size)
		var closev, x float64
		var cent []float64 = make([]float64, classes*layer1_size)
		for a := 0; a < vocab_size; a++ {
			cl[a] = a % clcn
		}
		for a := 0; a < iter; a++ {
			for b := 0; b < clcn*layer1_size; b++ {
				cent[b] = 0
			}
			for b := 0; b < clcn; b++ {
				centcn[b] = 1
			}
			for c := 0; c < vocab_size; c++ {
				for d := 0; d < layer1_size; d++ {
					cent[layer1_size*cl[c]+d] += syn0[c*layer1_size+d]
				}
				centcn[cl[c]]++
			}
			for b := 0; b < clcn; b++ {
				closev = 0
				for c := 0; c < layer1_size; c++ {
					cent[layer1_size*b+c] /= float64(centcn[b])
					closev += cent[layer1_size*b+c] * cent[layer1_size*b+c]
				}
				closev = math.Sqrt(closev)
				for c := 0; c < layer1_size; c++ {
					cent[layer1_size*b+c] /= closev
				}
			}
			for c := 0; c < vocab_size; c++ {
				closev = -10
				closeid = 0
				for d := 0; d < clcn; d++ {
					x = 0
					for b := 0; b < layer1_size; b++ {
						x += cent[layer1_size*d+b] * syn0[c*layer1_size+b]
					}
					if x > closev {
						closev = x
						closeid = d
					}
				}
				cl[c] = closeid
			}
		}
		// Save the K-means classes
		for a := 0; a < vocab_size; a++ {
			fmt.Fprintf(fo, "%c %d\n", vocab[a].char, cl[a])
		}
	}
	fo.Flush()
}

func ArgPos(str string, args []string) int {
	var a int
	for a = 1; a < len(args); a++ {
		if str == args[a] {
			if a == len(args)-1 {
				fmt.Fprintf(os.Stderr, "Argument missing for %s\n", str)
				os.Exit(1)
			}
			return a
		}
	}
	return -1
}

func main() {
	args := os.Args
	var i int
	if len(args) == 1 {
		fmt.Fprintf(os.Stderr, "CHARACTER VECTOR estimation toolkit v 0.1c\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		fmt.Fprintf(os.Stderr, "Parameters for training:\n")
		fmt.Fprintf(os.Stderr, "\t-train <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse text data from <file> to train the model\n")
		fmt.Fprintf(os.Stderr, "\t-output <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse <file> to save the resulting character vectors / character clusters\n")
		fmt.Fprintf(os.Stderr, "\t-size <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet size of character vectors; default is 100\n")
		fmt.Fprintf(os.Stderr, "\t-window <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet max skip length between characters; default is 5\n")
		fmt.Fprintf(os.Stderr, "\t-sample <float>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet threshold for occurrence of characters. Those that appear with higher frequency in the training data\n")
		fmt.Fprintf(os.Stderr, "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n")
		fmt.Fprintf(os.Stderr, "\t-hs <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse Hierarchical Softmax; default is 0 (not used)\n")
		fmt.Fprintf(os.Stderr, "\t-negative <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n")
		fmt.Fprintf(os.Stderr, "\t-threads <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse <int> threads (default 12)\n")
		fmt.Fprintf(os.Stderr, "\t-iter <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tRun more training iterations (default 5)\n")
		fmt.Fprintf(os.Stderr, "\t-min-count <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tThis will discard characters that appear less than <int> times; default is 5\n")
		fmt.Fprintf(os.Stderr, "\t-alpha <float>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n")
		fmt.Fprintf(os.Stderr, "\t-classes <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tOutput character classes rather than character vectors; default number of classes is 0 (vectors are written)\n")
		fmt.Fprintf(os.Stderr, "\t-debug <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSet the debug mode (default = 2 = more info during training)\n")
		fmt.Fprintf(os.Stderr, "\t-binary <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n")
		fmt.Fprintf(os.Stderr, "\t-save-vocab <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tThe vocabulary will be saved to <file>\n")
		fmt.Fprintf(os.Stderr, "\t-read-vocab <file>\n")
		fmt.Fprintf(os.Stderr, "\t\tThe vocabulary will be read from <file>, not constructed from the training data\n")
		fmt.Fprintf(os.Stderr, "\t-cbow <int>\n")
		fmt.Fprintf(os.Stderr, "\t\tUse the continuous bag of characters model; default is 1 (use 0 for skip-gram model)\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "./char2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n")
		return
	}
	output_file = ""
	save_vocab_file = ""
	read_vocab_file = ""
	if i := ArgPos("-size", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		layer1_size = int(v)
	}
	if i := ArgPos("-train", args); i > 0 {
		train_file = args[i+1]
	}
	if i := ArgPos("-save-vocab", args); i > 0 {
		save_vocab_file = args[i+1]
	}
	if i := ArgPos("-read-vocab", args); i > 0 {
		read_vocab_file = args[i+1]
	}
	if i := ArgPos("-debug", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		debug_mode = int(v)
	}
	if i := ArgPos("-binary", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		binaryf = int(v)
	}
	if i := ArgPos("-cbow", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		cbow = int(v)
	}
	if cbow != 0 {
		alpha = 0.05
	}
	if i := ArgPos("-alpha", args); i > 0 {
		v, _ := strconv.ParseFloat(args[i+1], 64)
		alpha = float64(v)
	}
	if i := ArgPos("-output", args); i > 0 {
		output_file = args[i+1]
	}
	if i := ArgPos("-window", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		window = int(v)
	}
	if i := ArgPos("-sample", args); i > 0 {
		v, _ := strconv.ParseFloat(args[i+1], 64)
		sample = float64(v)
	}
	if i := ArgPos("-hs", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		hs = int(v)
	}
	if i := ArgPos("-negative", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		negative = int(v)
	}
	fmt.Fprintf(os.Stderr, "negative: %d\n", negative)
	if i := ArgPos("-threads", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		num_threads = int(v)
	}
	if i := ArgPos("-iter", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		iter = int(v)
	}
	if i := ArgPos("-min-count", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		min_count = v
	}
	if i := ArgPos("-classes", args); i > 0 {
		v, _ := strconv.ParseInt(args[i+1], 10, 64)
		classes = int(v)
	}
	vocab = make([]vocab_char, vocab_max_size)
	vocab_hash = map[rune]int{}
	expTable = make([]float64, EXP_TABLE_SIZE+1)
	for i = 0; i < EXP_TABLE_SIZE; i++ {
		expTable[i] = math.Exp((float64(i)/float64(EXP_TABLE_SIZE)*2 - 1) * MAX_EXP) // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1)                                // Precompute f(x) = x / (x + 1)
	}
	TrainModel()
}
