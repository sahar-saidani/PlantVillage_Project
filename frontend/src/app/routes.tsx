import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Preprocessing } from "./pages/Preprocessing";
import { Segmentation } from "./pages/Segmentation";
import { FeatureExtraction } from "./pages/FeatureExtraction";
import { MLClassification } from "./pages/MLClassification";
import { DeepLearning } from "./pages/DeepLearning";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Dashboard },
      { path: "preprocessing", Component: Preprocessing },
      { path: "segmentation", Component: Segmentation },
      { path: "feature-extraction", Component: FeatureExtraction },
      { path: "classification", Component: MLClassification },
      { path: "deep-learning", Component: DeepLearning },
    ],
  },
]);
