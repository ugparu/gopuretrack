// Package lap solves the linear assignment problem using Jonker-Volgenant.
// The solver operates on square cost matrices; rectangular inputs are
// padded here to square by adding dummy rows/columns.
package lap

// Solver owns the scratch buffers used by one LAP solve. Reusing a Solver
// across calls amortizes the cost-matrix padding, the assignment output
// arrays, and the JV internal working memory — which otherwise dominates
// the allocation profile of real-time trackers that invoke the LAP
// several times per frame.
//
// A Solver is not safe for concurrent use: each goroutine that needs to
// solve LAPs should own its own instance. The zero value is usable.
type Solver struct {
	// padded n×n cost matrix: cRows[i] is a view into cFlat.
	cFlat []float64
	cRows [][]float64

	// JV outputs (length n), kept between calls so the allocator can be
	// skipped when subsequent solves are the same size or smaller.
	xIdxs []int
	yIdxs []int

	// JV internal scratch (length n).
	freeRows []int
	v        []float64
	unique   []bool
	cols     []int
	pred     []int
	d        []float64
}

// ensure grows every scratch buffer to at least length n, reallocating
// only the buffers that fall short. After this call, all buffer slices
// have length ≥ n.
func (s *Solver) ensure(n int) {
	if cap(s.cFlat) < n*n {
		s.cFlat = make([]float64, n*n)
	} else {
		s.cFlat = s.cFlat[:n*n]
	}
	if cap(s.cRows) < n {
		s.cRows = make([][]float64, n)
	} else {
		s.cRows = s.cRows[:n]
	}
	for i := range n {
		s.cRows[i] = s.cFlat[i*n : (i+1)*n : (i+1)*n]
	}
	growInts := func(buf []int) []int {
		if cap(buf) >= n {
			return buf[:n]
		}
		return make([]int, n)
	}
	growFloats := func(buf []float64) []float64 {
		if cap(buf) >= n {
			return buf[:n]
		}
		return make([]float64, n)
	}
	s.xIdxs = growInts(s.xIdxs)
	s.yIdxs = growInts(s.yIdxs)
	s.freeRows = growInts(s.freeRows)
	s.cols = growInts(s.cols)
	s.pred = growInts(s.pred)
	s.v = growFloats(s.v)
	s.d = growFloats(s.d)
	if cap(s.unique) >= n {
		s.unique = s.unique[:n]
	} else {
		s.unique = make([]bool, n)
	}
}

// Solve returns an optimal assignment between rows (e.g. tracks) and cols
// (e.g. detections) given a cost matrix. Pairs whose cost exceeds limit
// are rejected — the padded "dummy" rows and columns are filled with
// cost = limit/2, so a real assignment is only kept when two of them sum
// to less than limit.
//
// The returned tuple is (matches, unmatched row indices, unmatched column
// indices), where each match is a (row, col) pair. The returned slices
// are freshly allocated and do not alias the Solver's internal buffers.
func (s *Solver) Solve(rows, cols int, costs [][]float64, limit float64) ([][2]int, []int, []int) {
	if rows == 0 {
		unmatchedY := make([]int, 0, cols)
		for i := range cols {
			unmatchedY = append(unmatchedY, i)
		}
		return nil, nil, unmatchedY
	}
	if cols == 0 {
		unmatchedX := make([]int, 0, rows)
		for i := range rows {
			unmatchedX = append(unmatchedX, i)
		}
		return nil, unmatchedX, nil
	}

	n := rows + cols
	s.ensure(n)

	// Pad to a square n×n matrix. The top-left block is the real cost
	// matrix; the bottom-right block is all zeros so unused dummy
	// rows/columns match each other for free; the remaining two blocks
	// are filled with limit/2 so that a dummy-real pairing costs exactly
	// limit/2 — rejecting a real match via a dummy only becomes cheaper
	// than keeping it once the real cost crosses limit.
	half := limit / 2
	for i := range n {
		row := s.cRows[i]
		if i < rows {
			src := costs[i]
			for j := range cols {
				row[j] = src[j]
			}
			for j := cols; j < n; j++ {
				row[j] = half
			}
		} else {
			for j := range cols {
				row[j] = half
			}
			for j := cols; j < n; j++ {
				row[j] = 0
			}
		}
	}

	xIdxs := s.xIdxs[:n]
	yIdxs := s.yIdxs[:n]
	lapjvInternal(n, s.cRows[:n], xIdxs, yIdxs,
		s.freeRows[:n], s.v[:n], s.unique[:n],
		s.cols[:n], s.pred[:n], s.d[:n])

	// Any assignment that lands on a padded dummy column/row is a reject.
	// Trim the arrays back to the original rectangular shape.
	var matches [][2]int
	var unmatchedX, unmatchedY []int

	for i := range rows {
		x := xIdxs[i]
		if x >= 0 && x < cols {
			matches = append(matches, [2]int{i, x})
		} else {
			unmatchedX = append(unmatchedX, i)
		}
	}
	for j := range cols {
		y := yIdxs[j]
		if y < 0 || y >= rows {
			unmatchedY = append(unmatchedY, j)
		}
	}

	return matches, unmatchedX, unmatchedY
}

// SolveLinearAssignmentProblem is the allocation-heavy convenience form
// of Solver.Solve: it creates a throwaway Solver for every call. Code on
// a hot path should hold a Solver instance instead.
func SolveLinearAssignmentProblem(rows, cols int, costs [][]float64, limit float64) ([][2]int, []int, []int) {
	var s Solver
	return s.Solve(rows, cols, costs, limit)
}
