package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Data - входные данные и результат.
type Data struct {
	In  []float64
	Out float64
}

func main() {
	randSrc := rand.NewSource(time.Now().Unix())
	custRand := rand.New(randSrc)

	weights := map[string]float64{
		"i1_h1":   custRand.Float64() / 100,
		"i2_h1":   custRand.Float64() / 100,
		"bias_h1": custRand.Float64() / 100,
		"i1_h2":   custRand.Float64() / 100,
		"i2_h2":   custRand.Float64() / 100,
		"bias_h2": custRand.Float64() / 100,
		"h1_o1":   custRand.Float64() / 100,
		"h2_o1":   custRand.Float64() / 100,
		"bias_o1": custRand.Float64() / 100,
	}
	data := []*Data{
		&Data{In: []float64{0, 0}, Out: 0},
		&Data{In: []float64{1, 0}, Out: 1},
		&Data{In: []float64{0, 1}, Out: 1},
		&Data{In: []float64{1, 1}, Out: 0},
	}

	for i := 0; i < 10000; i++ {
		applyTrainUpdate(weights, train(weights, data))
	}
	outputResults(weights, data)
	fmt.Println("Error:", calculateResults(weights, data))
}

func activationSigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func derivativeSigmoid(x float64) float64 {
	fx := activationSigmoid(x)
	return fx * (1 - fx)
}

func nn(weights map[string]float64, i1, i2 float64) float64 {
	h1In := weights["i1_h1"]*i1 +
		weights["i2_h1"]*i2 +
		weights["bias_h1"]
	h1 := activationSigmoid(h1In)

	h2In := weights["i1_h2"]*i1 +
		weights["i2_h2"]*i2 +
		weights["bias_h2"]
	h2 := activationSigmoid(h2In)

	o1In := weights["h1_o1"]*h1 +
		weights["h2_o1"]*h2 +
		weights["bias_o1"]
	o1 := activationSigmoid(o1In)

	return o1
}

func train(weights map[string]float64, data []*Data) map[string]float64 {
	weightDeltas := map[string]float64{
		"i1_h1":   0,
		"i2_h1":   0,
		"bias_h1": 0,
		"i1_h2":   0,
		"i2_h2":   0,
		"bias_h2": 0,
		"h1_o1":   0,
		"h2_o1":   0,
		"bias_o1": 0,
	}

	for _, item := range data {
		h1In := weights["i1_h1"]*item.In[0] +
			weights["i2_h1"]*item.In[1] +
			weights["bias_h1"]
		h1 := activationSigmoid(h1In)

		h2In := weights["i1_h2"]*item.In[0] +
			weights["i2_h2"]*item.In[1] +
			weights["bias_h2"]
		h2 := activationSigmoid(h2In)

		o1In := weights["h1_o1"]*h1 +
			weights["h2_o1"]*h2 +
			weights["bias_o1"]

		o1 := activationSigmoid(o1In)

		delta := item.Out - o1
		o1Delta := delta * derivativeSigmoid(o1In)

		weightDeltas["h1_o1"] += h1 * o1Delta
		weightDeltas["h2_o1"] += h2 * o1Delta
		weightDeltas["bias_o1"] += o1Delta

		h1Delta := o1Delta * derivativeSigmoid(h1In)
		h2Delta := o1Delta * derivativeSigmoid(h2In)

		weightDeltas["i1_h1"] += item.In[0] * h1Delta
		weightDeltas["i2_h1"] += item.In[1] * h1Delta
		weightDeltas["bias_h1"] += h1Delta

		weightDeltas["i1_h2"] += item.In[0] * h2Delta
		weightDeltas["i2_h2"] += item.In[1] * h2Delta
		weightDeltas["bias_h2"] += h2Delta
	}

	return weightDeltas
}

func applyTrainUpdate(weights map[string]float64, weightDeltas map[string]float64) {
	for k := range weights {
		weights[k] += weightDeltas[k]
	}
}

func outputResults(weights map[string]float64, data []*Data) {
	for _, item := range data {
		fmt.Printf("%f XOR %f => %f (expected %f)\n", item.In[0], item.In[1],
			nn(weights, item.In[0], item.In[1]), item.Out)
	}
}

func calculateResults(weights map[string]float64, data []*Data) float64 {
	sum := 0.0
	for _, item := range data {
		tmp := nn(weights, item.In[0], item.In[1])
		sum += tmp * tmp
	}
	return sum / float64(len(data))
}
