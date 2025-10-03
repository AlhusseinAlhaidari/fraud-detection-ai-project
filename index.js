const tf = require("@tensorflow/tfjs-node");
const express = require("express");

const app = express();
const port = 3002;

app.use(express.json());

let model;

async function createAndTrainModel() {
  // Generate synthetic data for demonstration
  // Features: amount, location_risk (0-1), transaction_type (0-1), time_of_day (0-1)
  // Labels: 0 for legitimate, 1 for fraudulent
  const numSamples = 1000;
  const numFeatures = 4;

  const generateData = (count) => {
    const data = [];
    const labels = [];
    for (let i = 0; i < count; i++) {
      const amount = Math.random() * 1000; // Transaction amount
      const locationRisk = Math.random(); // Higher value means higher risk
      const transactionType = Math.round(Math.random()); // 0 or 1
      const timeOfDay = Math.random(); // 0 to 1 (e.g., normalized hours)

      let isFraud = 0;
      // Simple rule for synthetic fraud: high amount, high location risk, specific transaction type
      if (amount > 700 && locationRisk > 0.8 && transactionType === 1) {
        isFraud = 1;
      } else if (amount > 500 && locationRisk > 0.9) {
        isFraud = 1;
      } else if (locationRisk < 0.2 && amount < 100) {
        isFraud = 0; // Very low risk transactions are legitimate
      } else if (Math.random() < 0.05) { // Introduce some random fraud
        isFraud = 1;
      }

      data.push([amount / 1000, locationRisk, transactionType, timeOfDay]); // Normalize amount
      labels.push(isFraud);
    }
    return { data: tf.tensor2d(data), labels: tf.tensor2d(labels, [count, 1]) };
  };

  const { data: xTrain, labels: yTrain } = generateData(numSamples * 0.8);
  const { data: xTest, labels: yTest } = generateData(numSamples * 0.2);

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numFeatures], units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("Training fraud detection model...");
  await model.fit(xTrain, yTrain, {
    epochs: 50,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
      },
    },
  });
  console.log("Model training complete.");
}

app.post("/detect", async (req, res) => {
  if (!model) {
    return res.status(503).send("Model not trained yet. Please wait.");
  }

  const { amount, location_risk, transaction_type, time_of_day } = req.body;

  if (amount === undefined || location_risk === undefined || transaction_type === undefined || time_of_day === undefined) {
    return res.status(400).send("Missing transaction parameters.");
  }

  const inputTensor = tf.tensor2d([[amount / 1000, location_risk, transaction_type, time_of_day]]);
  const prediction = model.predict(inputTensor);
  const fraudProbability = prediction.dataSync()[0];
  const isFraud = fraudProbability > 0.5; // Threshold for fraud detection

  res.json({
    amount,
    location_risk,
    transaction_type,
    time_of_day,
    fraud_probability: fraudProbability.toFixed(4),
    is_fraud: isFraud,
    message: isFraud ? "Potential Fraudulent Transaction" : "Legitimate Transaction",
  });
});

app.listen(port, async () => {
  console.log(`Fraud Detection API listening at http://localhost:${port}`);
  console.log("Initializing and training model...");
  await createAndTrainModel();
  console.log("Model ready for predictions.");
  console.log("Try sending a POST request to /detect with a JSON body like:");
  console.log("{ \"amount\": 850, \"location_risk\": 0.9, \"transaction_type\": 1, \"time_of_day\": 0.7 }");
});

console.log("Fraud Detection project setup complete. Run `npm install` and then `node index.js` to start the server.");

