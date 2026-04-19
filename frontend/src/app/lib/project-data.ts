export type DatasetSummary = {
  num_images: number;
  classes: string[];
  metadata_path: string;
};

export type SegmentationBenchmark = {
  num_images_evaluated: number;
  per_class_limit: number;
  best_method: string | null;
  methods: Record<
    string,
    {
      description: string;
      quality_score: number;
      green_contrast: number;
      largest_component_ratio: number;
      edge_alignment: number;
      area_ratio: number;
    }
  >;
  ranking: Array<{ method: string; quality_score: number }>;
  notes: string[];
};

export type ModelMetrics = {
  accuracy: number;
  macro_f1: number;
  per_class: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }
  >;
  confusion_matrix?: number[][];
};

export type ComparisonSummary = {
  classes: string[];
  classical: {
    best_model: string;
    validation_results: Record<
      string,
      {
        accuracy: number;
        precision?: number;
        recall?: number;
        f1?: number;
        train_time_sec?: number;
      }
    >;
    test_accuracy: number | null;
    test_macro_f1: number | null;
    test_precision_macro: number | null;
    test_recall_macro: number | null;
  };
  deep_learning: {
    model: string;
    test_accuracy: number;
    test_macro_f1: number;
    best_val_acc: number;
  };
  modern_pipeline?: {
    advanced_preprocessing?: string[];
    advanced_segmentation?: SegmentationBenchmark | null;
  };
  delta_accuracy_deep_minus_classical: number | null;
  best_overall: string;
};

export type TrainingSummary = {
  selected_classes: string[];
  image_size: number;
  batch_size: number;
  epochs: number;
  best_val_acc: number;
  test_accuracy: number;
  macro_avg_f1: number;
  history: Array<Record<string, number>>;
};

async function fetchJson<T>(path: string): Promise<T | null> {
  const response = await fetch(path);
  if (!response.ok) {
    return null;
  }
  return (await response.json()) as T;
}

export async function loadProjectData() {
  const [
    datasetSummary,
    segmentationBenchmark,
    comparisonSummary,
    classicalSummary,
    svmMetrics,
    randomForestMetrics,
    trainingSummary,
    deepMetrics,
  ] = await Promise.all([
    fetchJson<DatasetSummary>("/project-data/dataset_summary.json"),
    fetchJson<SegmentationBenchmark>("/project-data/segmentation_benchmark.json"),
    fetchJson<ComparisonSummary>("/project-data/comparison_summary.json"),
    fetchJson<Record<string, unknown>>("/project-data/classical_clean_summary.json"),
    fetchJson<ModelMetrics>("/project-data/svm_metrics.json"),
    fetchJson<ModelMetrics>("/project-data/random_forest_metrics.json"),
    fetchJson<TrainingSummary>("/project-data/training_summary.json"),
    fetchJson<ModelMetrics>("/project-data/deep_metrics.json"),
  ]);

  return {
    datasetSummary,
    segmentationBenchmark,
    comparisonSummary,
    classicalSummary,
    svmMetrics,
    randomForestMetrics,
    trainingSummary,
    deepMetrics,
    exampleImages: [
      "/project-data/examples/Tomato_healthy_sample_1.png",
      "/project-data/examples/Tomato_Early_blight_sample_1.png",
      "/project-data/examples/Tomato_Late_blight_sample_1.png",
      "/project-data/examples/Tomato_Bacterial_spot_sample_1.png",
    ],
    labeledExamples: [
      { label: "Healthy sample", src: "/project-data/examples/Tomato_healthy_sample_1.png" },
      { label: "Early blight sample", src: "/project-data/examples/Tomato_Early_blight_sample_1.png" },
      { label: "Late blight sample", src: "/project-data/examples/Tomato_Late_blight_sample_1.png" },
      { label: "Bacterial spot sample", src: "/project-data/examples/Tomato_Bacterial_spot_sample_1.png" },
    ],
    confusionImages: {
      svm: "/project-data/svm_confusion_matrix.png",
      randomForest: "/project-data/random_forest_confusion_matrix.png",
      deep: "/project-data/deep_confusion_matrix.png",
      colab: "/project-data/colab_confusion_matrix.png",
    },
  };
}
