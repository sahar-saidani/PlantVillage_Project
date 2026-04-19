import { useEffect, useState } from "react";
import { loadProjectData } from "./project-data";

export function useProjectData() {
  const [data, setData] = useState<Awaited<ReturnType<typeof loadProjectData>> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    loadProjectData()
      .then((payload) => {
        if (active) {
          setData(payload);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return { data, loading };
}
