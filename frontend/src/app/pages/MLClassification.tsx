import React, { useState } from "react";
import { Upload, TrendingUp } from "lucide-react";

const models = [
  { id: "svm", name: "SVM (Support Vector Machine)" },
  { id: "rf", name: "Random Forest" },
  { id: "knn", name: "K-Nearest Neighbors" },
];

const metrics = [
  { name: "Accuracy", value: 94.2, color: "emerald" },
  { name: "Precision", value: 93.8, color: "blue" },
  { name: "Recall", value: 92.5, color: "amber" },
  { name: "F1-Score", value: 93.1, color: "rose" },
];

const confusionMatrix = [
  [89, 3, 2, 1, 0],
  [2, 78, 4, 2, 1],
  [1, 3, 82, 3, 2],
  [0, 2, 4, 75, 1],
  [1, 1, 2, 2, 88],
];

const classes = ["Healthy", "Rust", "Blight", "Mildew", "Spot"];

const predictions = [
  { class: "Healthy", confidence: 12 },
  { class: "Rust", confidence: 8 },
  { class: "Blight", confidence: 85 },
  { class: "Mildew", confidence: 18 },
  { class: "Spot", confidence: 22 },
];

export function MLClassification() {
  const [selectedModel, setSelectedModel] = useState("rf");
  const [showPrediction, setShowPrediction] = useState(false);

  const maxConfusion = Math.max(...confusionMatrix.flat());

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Machine Learning Classification</h1>
        <p className="text-gray-400">Traditional ML models for disease detection</p>
      </div>

      {/* Model Selector */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Select Model</h2>
        <div className="space-y-3">
          {models.map((model) => (
            <label
              key={model.id}
              className="flex items-center gap-3 p-4 bg-gray-900/50 rounded-lg cursor-pointer hover:bg-gray-900 transition-all"
            >
              <input
                type="radio"
                name="model"
                value={model.id}
                checked={selectedModel === model.id}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-4 h-4 text-emerald-500 accent-emerald-500"
              />
              <span className={selectedModel === model.id ? "text-emerald-500 font-semibold" : ""}>
                {model.name}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <div key={metric.name} className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
            <p className="text-sm text-gray-400 mb-2">{metric.name}</p>
            <h3 className="text-3xl font-bold mb-2">{metric.value}%</h3>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`bg-${metric.color}-500 h-2 rounded-full transition-all`}
                style={{ width: `${metric.value}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Confusion Matrix */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Confusion Matrix</h2>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="grid grid-cols-6 gap-2">
              <div></div>
              {classes.map((cls) => (
                <div key={cls} className="text-center text-sm text-gray-400 font-semibold p-2">
                  {cls}
                </div>
              ))}
              {confusionMatrix.map((row, i) => (
                <React.Fragment key={`row-${i}`}>
                  <div className="flex items-center justify-end text-sm text-gray-400 font-semibold p-2">
                    {classes[i]}
                  </div>
                  {row.map((value, j) => {
                    const intensity = value / maxConfusion;
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        className="aspect-square rounded-lg flex items-center justify-center text-sm font-semibold transition-all hover:scale-105"
                        style={{
                          backgroundColor: i === j
                            ? `rgba(16, 185, 129, ${0.2 + intensity * 0.8})`
                            : `rgba(239, 68, 68, ${0.1 + intensity * 0.4})`,
                          color: intensity > 0.5 ? "#fff" : "#9ca3af"
                        }}
                      >
                        {value}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Single Image Prediction */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Predict Single Image</h2>
        <div
          onClick={() => setShowPrediction(true)}
          className="border-2 border-dashed border-gray-700 hover:border-emerald-500/50 rounded-lg p-8 cursor-pointer transition-all mb-6"
        >
          <div className="flex flex-col items-center text-center">
            <Upload className="w-12 h-12 text-emerald-500 mb-3" />
            <p className="text-sm text-gray-400">Upload image for prediction</p>
          </div>
        </div>

        {showPrediction && (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <div>
                <p className="text-sm text-gray-400 mb-1">Predicted Class</p>
                <h3 className="text-2xl font-bold text-emerald-500">Blight</h3>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-400 mb-1">Confidence</p>
                <h3 className="text-2xl font-bold text-emerald-500">85%</h3>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-sm text-gray-400 font-semibold">Class Probabilities</p>
              {predictions.map((pred) => (
                <div key={pred.class} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className={pred.confidence > 50 ? "text-emerald-500 font-semibold" : "text-gray-400"}>
                      {pred.class}
                    </span>
                    <span className={pred.confidence > 50 ? "text-emerald-500 font-semibold" : "text-gray-400"}>
                      {pred.confidence}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        pred.confidence > 50 ? "bg-emerald-500" : "bg-gray-600"
                      }`}
                      style={{ width: `${pred.confidence}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
