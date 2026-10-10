import type { GitDiffViewerItem } from "../git/components/GitDiffViewer.types";

const FILE_HEADER = /^diff --git /gm;

function normalizePatchPath(rawPath: string): string | null {
  const trimmed = rawPath.trim();
  if (!trimmed || trimmed === "/dev/null") {
    return null;
  }

  const unquoted = trimmed.startsWith('"') && trimmed.endsWith('"')
    ? trimmed.slice(1, -1).replace(/\\([\\"])/g, "$1")
    : trimmed;

  return unquoted.replace(/^(?:a|b)\//, "");
}

function metadataPath(diff: string, prefix: "--- " | "+++ "): string | null {
  const line = diff
    .split("\n")
    .find((candidate) => candidate.startsWith(prefix));
  if (!line) {
    return null;
  }

  const rawPath = line.slice(prefix.length).split("\t", 1)[0] ?? "";
  return normalizePatchPath(rawPath);
}

function renamePath(diff: string, prefix: "rename from " | "rename to "): string | null {
  const line = diff
    .split("\n")
    .find((candidate) => candidate.startsWith(prefix));
  return line ? normalizePatchPath(line.slice(prefix.length)) : null;
}

function headerPaths(diff: string): { oldPath: string | null; newPath: string | null } {
  const firstLine = diff.split("\n", 1)[0] ?? "";
  const match = /^diff --git a\/(.+) b\/(.+)$/.exec(firstLine);
  return {
    oldPath: match?.[1] ?? null,
    newPath: match?.[2] ?? null,
  };
}

function adaptFileDiff(diff: string): GitDiffViewerItem | null {
  const header = headerPaths(diff);
  const oldPath = renamePath(diff, "rename from ")
    ?? metadataPath(diff, "--- ")
    ?? header.oldPath;
  const newPath = renamePath(diff, "rename to ")
    ?? metadataPath(diff, "+++ ")
    ?? header.newPath;

  const isRename = /^rename (?:from|to) /m.test(diff);
  const isAdded = /^new file mode /m.test(diff) || /^--- \/dev\/null(?:\t.*)?$/m.test(diff);
  const isDeleted = /^deleted file mode /m.test(diff) || /^\+\+\+ \/dev\/null(?:\t.*)?$/m.test(diff);
  const path = isDeleted ? oldPath : newPath;

  if (!path) {
    return null;
  }

  return {
    path,
    status: isRename ? "R" : isAdded ? "A" : isDeleted ? "D" : "M",
    diff,
  };
}

export function parseRemoteDiff(patch: string): GitDiffViewerItem[] {
  const normalizedPatch = patch.replace(/\r\n?/g, "\n");
  const starts = Array.from(normalizedPatch.matchAll(FILE_HEADER), (match) => match.index);

  return starts.flatMap((start, index) => {
    const end = starts[index + 1] ?? normalizedPatch.length;
    const fileDiff = normalizedPatch.slice(start, end).replace(/\n+$/, "");
    const item = adaptFileDiff(fileDiff);
    return item ? [item] : [];
  });
}

export const unifiedGitPatchToGitDiffViewerItems = parseRemoteDiff;
