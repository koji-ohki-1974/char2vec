package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
)

const max_size int = 2000 // max length of strings
const N int = 40          // number of closest characters that will be shown
const max_w int = 50      // max length of vocabulary entries

func main() {
	args := os.Args
	var st1 string
	var bestw []rune = make([]rune, N)
	var dist, length float64
	var bestd []float64 = make([]float64, N)
	var vec []float64 = make([]float64, max_size)
	var chars, size, a, b, c, d, cn int
	var bi []int = make([]int, 100)
	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: ./char-analogy <FILE>\nwhere FILE contains character projections in the BINARY FORMAT\n")
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
		for a = 0; a < N; a++ {
			bestd[a] = 0
		}
		for a = 0; a < N; a++ {
			bestw[a] = 0
		}
		fmt.Printf("Enter three characters (EXIT to break): ")
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
		for _, ch := range st1 {
			for b = 0; b < chars; b++ {
				if vocab[b] == ch {
					break
				}
			}
			if b == chars {
				b = 0
			}
			bi[a] = b
			fmt.Printf("\nCharacter: %c  Position in vocabulary: %d\n", ch, bi[a])
			if b == 0 {
				fmt.Printf("Out of dictionary character!\n")
				break
			}
			a++
		}
		cn = a
		if cn < 3 {
			fmt.Printf("Only %d characters were entered.. three characters are needed at the input to perform the calculation\n", cn)
			continue
		}
		if b == 0 {
			continue
		}
		fmt.Printf("\n      Character         Distance\n------------------------------------------------------------------------\n")
		for a = 0; a < size; a++ {
			vec[a] = M[a+bi[1]*size] - M[a+bi[0]*size] + M[a+bi[2]*size]
		}
		length = 0
		for a = 0; a < size; a++ {
			length += vec[a] * vec[a]
		}
		length = math.Sqrt(length)
		for a = 0; a < size; a++ {
			vec[a] /= length
		}
		for a = 0; a < N; a++ {
			bestd[a] = 0
		}
		for a = 0; a < N; a++ {
			bestw[a] = 0
		}
		for c = 0; c < chars; c++ {
			if c == bi[0] {
				continue
			}
			if c == bi[1] {
				continue
			}
			if c == bi[2] {
				continue
			}
			a = 0
			for b := 0; b < cn; b++ {
				if bi[b] == c {
					a = 1
				}
			}
			if a == 1 {
				continue
			}
			dist = 0
			for a = 0; a < size; a++ {
				dist += vec[a] * M[a+c*size]
			}
			for a = 0; a < N; a++ {
				if dist > bestd[a] {
					for d = N - 1; d > a; d-- {
						bestd[d] = bestd[d-1]
						bestw[d] = bestw[d-1]
					}
					bestd[a] = dist
					bestw[a] = vocab[c]
					break
				}
			}
		}
		for a = 0; a < N; a++ {
			fmt.Printf("%10c\t\t%f\n", bestw[a], bestd[a])
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
