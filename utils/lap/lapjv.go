package lap

// Port of lapjv.cpp (Jonker-Volgenant linear assignment, dense matrix).
// The translation preserves the original function structure and variable
// names so the two implementations can be cross-checked side-by-side.
//
// All scratch buffers (freeRows, v, unique, cols, d, pred) are owned by
// the caller and passed in. Each is used as pure scratch: the solver
// does not assume incoming content and fully initializes what it needs
// at the top of each call.

const large = 1000000.0

// ccrrtDense: column-reduction and reduction transfer for a dense cost
// matrix. Returns the number of still-unassigned rows written to freeRows.
// unique is scratch of length ≥ n.
func ccrrtDense(n int, cost [][]float64, freeRows, x, y []int, v []float64, unique []bool) int {
	for i := range n {
		x[i] = -1
		v[i] = large
		y[i] = 0
	}
	for i := range n {
		row := cost[i]
		for j := range n {
			c := row[j]
			if c < v[j] {
				v[j] = c
				y[j] = i
			}
		}
	}
	for i := range n {
		unique[i] = true
	}
	// do { j--; ... } while (j > 0) with j initialized to n iterates
	// j = n-1 .. 0 inclusive.
	for j := n - 1; j >= 0; j-- {
		i := y[j]
		if x[i] < 0 {
			x[i] = j
		} else {
			unique[i] = false
			y[j] = -1
		}
	}
	nFreeRows := 0
	for i := range n {
		if x[i] < 0 {
			freeRows[nFreeRows] = i
			nFreeRows++
		} else if unique[i] {
			j := x[i]
			min := large
			row := cost[i]
			for j2 := range n {
				if j2 == j {
					continue
				}
				c := row[j2] - v[j2]
				if c < min {
					min = c
				}
			}
			v[j] -= min
		}
	}
	return nFreeRows
}

// carrDense: augmenting row reduction for a dense cost matrix.
func carrDense(n int, cost [][]float64, nFreeRows int, freeRows, x, y []int, v []float64) int {
	current := 0
	newFreeRows := 0
	rrCnt := 0
	for current < nFreeRows {
		rrCnt++
		freeI := freeRows[current]
		current++
		row := cost[freeI]
		j1 := 0
		v1 := row[0] - v[0]
		j2 := -1
		v2 := large
		for j := 1; j < n; j++ {
			c := row[j] - v[j]
			if c < v2 {
				if c >= v1 {
					v2 = c
					j2 = j
				} else {
					v2 = v1
					v1 = c
					j2 = j1
					j1 = j
				}
			}
		}
		i0 := y[j1]
		v1New := v[j1] - (v2 - v1)
		v1Lowers := v1New < v[j1]
		if rrCnt < current*n {
			if v1Lowers {
				v[j1] = v1New
			} else if i0 >= 0 && j2 >= 0 {
				j1 = j2
				i0 = y[j2]
			}
			if i0 >= 0 {
				if v1Lowers {
					current--
					freeRows[current] = i0
				} else {
					freeRows[newFreeRows] = i0
					newFreeRows++
				}
			}
		} else {
			if i0 >= 0 {
				freeRows[newFreeRows] = i0
				newFreeRows++
			}
		}
		x[freeI] = j1
		y[j1] = freeI
	}
	return newFreeRows
}

// findDense: find columns with minimum d[j] and put them on the SCAN list.
func findDense(n, lo int, d []float64, cols []int) int {
	hi := lo + 1
	mind := d[cols[lo]]
	for k := hi; k < n; k++ {
		j := cols[k]
		if d[j] <= mind {
			if d[j] < mind {
				hi = lo
				mind = d[j]
			}
			cols[k] = cols[hi]
			cols[hi] = j
			hi++
		}
	}
	return hi
}

// scanDense: scan all columns in TODO starting from arbitrary column in
// SCAN and try to decrease d of the TODO columns using the SCAN column.
// Returns the closest free column index or -1 if none yet. *loPtr/*hiPtr
// are only updated when the function falls through the outer loop — on
// an early return the caller's lo/hi stay untouched, matching the C code.
func scanDense(n int, cost [][]float64, loPtr, hiPtr *int, d []float64, cols, pred, y []int, v []float64) int {
	lo := *loPtr
	hi := *hiPtr
	for lo != hi {
		j := cols[lo]
		lo++
		i := y[j]
		mind := d[j]
		row := cost[i]
		h := row[j] - v[j] - mind
		for k := hi; k < n; k++ {
			j2 := cols[k]
			credIJ := row[j2] - v[j2] - h
			if credIJ < d[j2] {
				d[j2] = credIJ
				pred[j2] = i
				if credIJ == mind {
					if y[j2] < 0 {
						return j2
					}
					cols[k] = cols[hi]
					cols[hi] = j2
					hi++
				}
			}
		}
	}
	*loPtr = lo
	*hiPtr = hi
	return -1
}

// findPathDense: single iteration of modified Dijkstra shortest path as
// described in the JV paper (dense matrix version). cols and d are scratch
// of length ≥ n; both are fully initialized at the top of this call.
// Returns the closest free column index.
func findPathDense(n int, cost [][]float64, startI int, y []int, v []float64, pred, cols []int, d []float64) int {
	lo, hi := 0, 0
	finalJ := -1
	nReady := 0
	startRow := cost[startI]
	for i := range n {
		cols[i] = i
		pred[i] = startI
		d[i] = startRow[i] - v[i]
	}
	for finalJ == -1 {
		if lo == hi {
			nReady = lo
			hi = findDense(n, lo, d, cols)
			for k := lo; k < hi; k++ {
				j := cols[k]
				if y[j] < 0 {
					finalJ = j
				}
			}
		}
		if finalJ == -1 {
			finalJ = scanDense(n, cost, &lo, &hi, d, cols, pred, y, v)
		}
	}
	mind := d[cols[lo]]
	for k := range nReady {
		j := cols[k]
		v[j] += d[j] - mind
	}
	return finalJ
}

// caDense: augment for a dense cost matrix. pred/cols are scratch of
// length ≥ n and are reused across every findPathDense call inside.
func caDense(n int, cost [][]float64, nFreeRows int, freeRows, x, y []int, v []float64, pred, cols []int, d []float64) {
	for idx := range nFreeRows {
		freeI := freeRows[idx]
		i := -1
		j := findPathDense(n, cost, freeI, y, v, pred, cols, d)
		for i != freeI {
			i = pred[j]
			y[j] = i
			j, x[i] = x[i], j
		}
	}
}

// lapjvInternal: solve dense LAP. n is the square matrix size. cost[i][j]
// is the cost of assigning row i to column j. On return, x[i] holds the
// column assigned to row i and y[j] the row assigned to column j.
// All scratch slices must have length ≥ n.
func lapjvInternal(n int, cost [][]float64, x, y []int, freeRows []int, v []float64, unique []bool, cols, pred []int, d []float64) {
	ret := ccrrtDense(n, cost, freeRows, x, y, v, unique)
	for i := 0; ret > 0 && i < 2; i++ {
		ret = carrDense(n, cost, ret, freeRows, x, y, v)
	}
	if ret > 0 {
		caDense(n, cost, ret, freeRows, x, y, v, pred, cols, d)
	}
}
