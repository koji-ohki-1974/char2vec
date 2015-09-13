package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

const max_size int = 2000 // max length of strings
const N int = 20          // number of closest characters that will be shown
const N_CUT int = 0
const max_w int = 50 // max length of vocabulary entries
const sample float64 = .5
const penalty float64 = .5

var window int = 8
var output_length int = 1000

func main() {
	args := os.Args
	var st1 string
	var besti []int = make([]int, N)
	var bestw []rune = make([]rune, N)
	var dist, length float64
	var bestd []float64 = make([]float64, N)
	var vec []float64 = make([]float64, max_size)
	var chars, size, a, b, c, d int
	var bi0 []int
	var bi []int

	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: ./char-writing <FILE>\nwhere FILE contains character projections in the BINARY FORMAT\n")
		os.Exit(0)
	}
	file_name := args[1]
	f, err := os.Open(file_name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Input file not found\n")
		os.Exit(-1)
	}
	defer f.Close()
	br := bufio.NewReader(f)
	fmt.Fscanf(br, "%d", &chars)
	fmt.Fprintf(os.Stderr, "characters: %d\n", chars)
	fmt.Fscanf(br, "%d", &size)
	fmt.Fprintf(os.Stderr, "size: %d\n", size)
	br.ReadByte()
	vocab := make([]rune, chars)
	M := make([]float64, chars*size)
	for b = 0; b < chars; b++ {
		vocab[b], _, err = br.ReadRune()
		br.ReadByte()
		err = binary.Read(br, binary.LittleEndian, M[b*size:b*size+size])
		failOnError(err, "Cannot read input file")
		length = 0
		for a = 0; a < size; a++ {
			length += M[a+b*size] * M[a+b*size]
		}
		length = math.Sqrt(length)
		for a = 0; a < size; a++ {
			M[a+b*size] /= length
		}
		br.ReadByte()
	}
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("\nEnter character or character sequence (EXIT to break): ")
		sf := scanner.Scan()
		if !sf {
			break
		}
		st1 = scanner.Text()
		if st1 == "EXIT" {
			break
		}
		a = 0
		b = 0
		c = 0
		n := 0
		bi0 = []int{}
		bi = make([]int, window)
		for _, ch := range st1 {
			for b = 0; b < chars; b++ {
				if vocab[b] == ch {
					break
				}
			}
			if b == chars {
				b = -1
			}
			bi0 = append(bi0, b)
			fmt.Printf("%c", ch)
			n++
		}

		bilen := 0
		bipos := 0
		lasti := 0
		for ; n < output_length; n++ {
			for a = 0; a < N; a++ {
				besti[a] = -1
			}
			for a = 0; a < N; a++ {
				bestd[a] = -1
			}
			for a = 0; a < N; a++ {
				bestw[a] = 0
			}
			for a = 0; a < size; a++ {
				vec[a] = 0
			}
			for _, v := range append(bi0, bi[:bilen]...) {
				if v == -1 {
					continue
				}
				if rand.Float64() < sample {
					for a = 0; a < size; a++ {
						vec[a] += M[a+v*size]
					}
				}
			}
			length = 0
			for a = 0; a < size; a++ {
				length += vec[a] * vec[a]
			}
			length = math.Sqrt(length)
			if length != 0 {
				for a = 0; a < size; a++ {
					vec[a] /= length
				}
			}
			for c = 0; c < chars; c++ {
				dist = 0
				for a = 0; a < size; a++ {
					dist += vec[a] * M[a+c*size]
				}
				for a = 0; a < N; a++ {
					if dist > bestd[a] {
						for d = N - 1; d > a; d-- {
							besti[d] = besti[d-1]
							bestd[d] = bestd[d-1]
							bestw[d] = bestw[d-1]
						}
						besti[a] = c
						bestd[a] = dist
						bestw[a] = vocab[c]
						break
					}
				}
			}
			sum := 0.
			penaltyf := rand.Float64() < penalty
			for a = N_CUT; a < N; a++ {
				if !penaltyf || lasti != besti[a] {
					sum += bestd[a]
				}
			}
			r := sum * rand.Float64()
			sum = 0.
			for a = N_CUT; a < N; a++ {
				if !penaltyf || lasti != besti[a] {
					sum += bestd[a]
				}
				if r < sum {
					fmt.Printf("%c", bestw[a])
					bi[bipos] = besti[a]
					lasti = besti[a]
					bipos = (bipos + 1) % window
					bilen++
					if window < bilen {
						bilen = window
					}
					break
				}
			}
		}
	}
	os.Exit(0)
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
		panic(fmt.Sprintf("%s: %s", msg, err))
	}
}
