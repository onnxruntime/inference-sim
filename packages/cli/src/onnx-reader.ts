import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import type { OnnxModelManifest } from "@inference-sim/core";
import { inspectOnnxModelBytes } from "@inference-sim/onnx-inspector";

export async function inspectOnnxModel(
  modelPath: string,
  metadata?: unknown,
): Promise<OnnxModelManifest> {
  const resolvedModelPath = resolve(modelPath);
  return inspectOnnxModelBytes({
    modelFileName: basename(modelPath),
    modelBytes: await readFile(resolvedModelPath),
    metadata,
    sha256: async (bytes) => (
      createHash("sha256").update(bytes).digest("hex")
    ),
  });
}
