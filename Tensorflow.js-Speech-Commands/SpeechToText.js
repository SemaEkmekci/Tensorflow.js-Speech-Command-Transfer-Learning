import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';
import axios from 'axios';

const NUM_FRAMES = 3;

function SpeechToText() {
  const [model, setModel] = useState(null);
  const [recognizer, setRecognizer] = useState(null);
  const [examples, setExamples] = useState([]);
  const [isListening, setIsListening] = useState(false);

  async function saveModelToServer(modelData) {
    try {
      const response = await fetch('http://localhost:3001/save-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(modelData)
      });

      if (response.ok) {
        console.log('Model successfully saved on the server.');
      } else {
        console.error('Failed to save the model on the server.');
      }
    } catch (error) {
      console.error('An error occurred while saving the model:', error);
    }
  }

  useEffect(() => {
    async function setup() {
      const loadedRecognizer = speechCommands.create('BROWSER_FFT');
      await loadedRecognizer.ensureModelLoaded();
      setRecognizer(loadedRecognizer);

      const loadedModel = buildModel();
      setModel(loadedModel);
    }

    setup();
  }, []);

  function buildModel() {
    const newModel = tf.sequential();
    newModel.add(tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 3],
      activation: 'relu',
      inputShape: [NUM_FRAMES, 232, 1]
    }));
    newModel.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
    newModel.add(tf.layers.flatten());
    newModel.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    const optimizer = tf.train.adam(0.01);
    newModel.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    return newModel;
  }

  function collect(label) {
    if (recognizer.isListening()) {
      return recognizer.stopListening();
    }
    if (label == null) {
      return;
    }
    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      setExamples(prevExamples => [...prevExamples, { vals, label }]);
      console.log(`${examples.length} examples collected`)
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
  }

  async function train() {
    const ys = tf.oneHot(examples.map(e => e.label), 3);
    const xsShape = [examples.length, NUM_FRAMES, 232, 1];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

    await model.fit(xs, ys, {
      batchSize: 16,
      epochs: 10,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`);
        }
      }
    });

    tf.dispose([xs, ys]);

    // const modelData = await model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
    //   return { data: await modelArtifacts.modelTopology, format: 'json' };
    // }));
    await model.save('downloads://my-model');
    // console.log(modelData);
    // console.log("Model istemci tarafında kayıt edildi.");
    // await saveModelToServer(modelData);
  }

  function toggleListening() {
    if (recognizer.isListening()) {
      recognizer.stopListening();
      setIsListening(false);
      return;
    }

    setIsListening(true);
    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, NUM_FRAMES, 232, 1]);
      const probs = model.predict(input);
      const predLabel = probs.argMax(1);
      await moveSlider(predLabel);
      tf.dispose([input, probs, predLabel]);
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
  }

  async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    console.log(label);
    if (label === 2) {
      return;
    }
    const delta = 0.1;
    const outputElement = document.getElementById('output');
    const prevValue = +outputElement.value;
    outputElement.value = prevValue + (label === 0 ? -delta : delta);
  }

  function normalize(x) {
    const mean = -100;
    const std = 10;
    return x.map(x => (x - mean) / std);
  }

  function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));
    return result;
  }

  return (
    <div className="App">
      <div id="console">
        <button id="sema" onMouseDown={() => collect(0)} onMouseUp={() => collect(null)}>sema</button>
        <button id="ekmekci" onMouseDown={() => collect(1)} onMouseUp={() => collect(null)}>ekmekci</button>
        <button id="merhaba" onMouseDown={() => collect(2)} onMouseUp={() => collect(null)}>merhaba</button>
        <br /><br />
        <button id="train" onClick={train}>Train and Save</button>
        <br /><br />
        <button id="listen" onClick={toggleListening}>
          {isListening ? 'Stop' : 'Listen'}
        </button>
        <br /><br />
        <input type="range" id="output" min="0" max="10" step="0.1" />
      </div>
    </div>
  );
}

export default SpeechToText;
