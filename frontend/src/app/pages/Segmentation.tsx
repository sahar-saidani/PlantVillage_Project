import { useState } from "react";
import { Image as ImageIcon, Check } from "lucide-react";

const segmentationMethods = [
  { id: "canny", name: "Canny Edges", quality: 94 },
  { id: "sobel", name: "Sobel", quality: 89 },
  { id: "hsv", name: "HSV Mask", quality: 96 },
];

export function Segmentation() {
  const [activeMethod, setActiveMethod] = useState("canny");

  const panels = [
    { label: "Original", gradient: "from-green-900 to-emerald-700", quality: 100 },
    { label: "Canny Edges", gradient: "from-gray-900 to-emerald-900/30", quality: 94 },
    { label: "Sobel", gradient: "from-gray-800 to-emerald-800/40", quality: 89 },
    { label: "HSV Mask", gradient: "from-emerald-900 to-yellow-900/50", quality: 96 },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Segmentation</h1>
        <p className="text-gray-400">Edge detection and region extraction</p>
      </div>

      {/* Method Toggle */}
      <div className="flex gap-3">
        {segmentationMethods.map((method) => (
          <button
            key={method.id}
            onClick={() => setActiveMethod(method.id)}
            className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${
              activeMethod === method.id
                ? "bg-emerald-500 text-white"
                : "bg-[#1a1d27] text-gray-400 hover:bg-gray-800"
            }`}
          >
            {activeMethod === method.id && <Check className="w-4 h-4" />}
            {method.name}
          </button>
        ))}
      </div>

      {/* Segmentation Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {panels.map((panel, index) => (
          <div
            key={panel.label}
            className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800 hover:border-emerald-500/30 transition-all"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">{panel.label}</h3>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Quality</span>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  panel.quality >= 95 ? "bg-emerald-500/10 text-emerald-500" :
                  panel.quality >= 90 ? "bg-amber-500/10 text-amber-500" :
                  "bg-blue-500/10 text-blue-500"
                }`}>
                  {panel.quality}%
                </span>
              </div>
            </div>
            <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden">
              <div className={`w-full h-full bg-gradient-to-br ${panel.gradient} flex items-center justify-center relative`}>
                <ImageIcon className="w-16 h-16 text-white/20" />
                {index > 0 && (
                  <div className="absolute inset-0 opacity-30">
                    <svg className="w-full h-full" viewBox="0 0 400 300">
                      {index === 1 && (
                        <>
                          <path d="M50,150 Q100,100 150,150 T250,150" stroke="#10b981" strokeWidth="2" fill="none" />
                          <path d="M100,200 L150,180 L200,190 L250,170" stroke="#10b981" strokeWidth="2" fill="none" />
                          <circle cx="150" cy="100" r="40" stroke="#10b981" strokeWidth="2" fill="none" />
                        </>
                      )}
                      {index === 2 && (
                        <>
                          <line x1="80" y1="80" x2="320" y2="80" stroke="#10b981" strokeWidth="1.5" />
                          <line x1="80" y1="150" x2="320" y2="150" stroke="#10b981" strokeWidth="1.5" />
                          <line x1="80" y1="220" x2="320" y2="220" stroke="#10b981" strokeWidth="1.5" />
                        </>
                      )}
                      {index === 3 && (
                        <>
                          <ellipse cx="200" cy="150" rx="120" ry="80" stroke="#10b981" strokeWidth="3" fill="none" />
                          <ellipse cx="200" cy="150" rx="60" ry="40" stroke="#10b981" strokeWidth="2" fill="none" />
                        </>
                      )}
                    </svg>
                  </div>
                )}
              </div>
            </div>
            {index > 0 && (
              <div className="mt-4 flex items-center justify-between text-xs">
                <span className="text-gray-400">Edge Detection</span>
                <span className="text-emerald-500 font-semibold">Active</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Method Details */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Algorithm Details</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {segmentationMethods.map((method) => (
            <div key={method.id} className="space-y-2">
              <h3 className="font-semibold text-emerald-500">{method.name}</h3>
              <div className="space-y-1 text-sm text-gray-400">
                <p>Quality Score: {method.quality}%</p>
                <p>Processing Time: {Math.floor(Math.random() * 200 + 100)}ms</p>
                <p>Edge Pixels: {Math.floor(Math.random() * 5000 + 10000)}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
