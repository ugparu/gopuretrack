# GoPureTrack

[![CI](https://github.com/ugparu/gopuretrack/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ugparu/gopuretrack/actions/workflows/ci.yml)
[![Lint](https://github.com/ugparu/gopuretrack/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/ugparu/gopuretrack/actions/workflows/lint.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ugparu/gopuretrack.svg)](https://pkg.go.dev/github.com/ugparu/gopuretrack)
[![Go Report Card](https://goreportcard.com/badge/github.com/ugparu/gopuretrack)](https://goreportcard.com/report/github.com/ugparu/gopuretrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

GoPureTrack is a high-performance object tracker written in Go. It is inspired by the [DeepSort tracker](https://github.com/nwojke/deep_sort) but incorporates significant modifications and improvements for enhanced performance and flexibility.

![MOT17-02 GIF](examples/MOT17-02.gif)

## Features

- **Efficient Go Implementation**: Optimized for high performance while leveraging Go with minimal cgo components.
- **Enhanced Object Tracking**: Built upon DeepSort principles, with changes to optimize accuracy and speed.
- **Real-Time Performance**: Capable of handling high-speed object tracking scenarios.
- **Customizable API**: Easily integrate into various applications with flexible configurations.

## Installation

To install GoPureTrack, you need to have Go installed on your system. Use the following command to install it:

```bash
go get github.com/ugparu/gopuretrack
```

## Usage

Here is a quick example to get started with GoPureTrack:

```go
package main

import (
	"fmt"
	"github.com/ugparu/gopuretrack/puretrack"
)

type Detection struct {
	XYXY  [4]float64
	score float64
}

func (d Detection) GetXYXY() [4]float64 {
	return d.XYXY
}

func (d Detection) GetScore() float64 {
	return d.score
}

func main() {
	tracker := puretrack.New[Detection](puretrack.BaseConfig)

	detections := []Detection{......}

	trackedObjects, removedTracks := tracker.Update(detections)
	fmt.Println("Tracked objects:", trackedObjects)
	fmt.Println("Removed tracks:", removedTracks)
}
```

## Contributions

We welcome contributions to GoPureTrack! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

## License

GoPureTrack is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.

## Acknowledgments

This project is inspired by the [DeepSort tracker](https://github.com/nwojke/deepsort). Special thanks to the open-source community for providing the foundation for this implementation.

