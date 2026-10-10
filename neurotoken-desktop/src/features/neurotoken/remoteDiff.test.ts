import { describe, expect, expectTypeOf, it } from "vitest";
import type { GitDiffViewerItem } from "../git/components/GitDiffViewer.types";
import { parseRemoteDiff } from "./remoteDiff";

describe("parseRemoteDiff", () => {
  it("adapts an added file", () => {
    const result = parseRemoteDiff(`diff --git a/src/new.ts b/src/new.ts
new file mode 100644
index 0000000..3b18e13
--- /dev/null
+++ b/src/new.ts
@@ -0,0 +1 @@
+export const value = 1;
`);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ path: "src/new.ts", status: "A" });
    expect(result[0]?.diff).toContain("+export const value = 1;");
    expectTypeOf(result).toEqualTypeOf<GitDiffViewerItem[]>();
  });

  it("adapts a modified file", () => {
    const result = parseRemoteDiff(`diff --git a/src/value.ts b/src/value.ts
index 1111111..2222222 100644
--- a/src/value.ts
+++ b/src/value.ts
@@ -1 +1 @@
-export const value = 1;
+export const value = 2;
`);

    expect(result[0]).toMatchObject({ path: "src/value.ts", status: "M" });
  });

  it("uses the old path for a deleted file", () => {
    const result = parseRemoteDiff(`diff --git a/src/old.ts b/src/old.ts
deleted file mode 100644
index 3b18e13..0000000
--- a/src/old.ts
+++ /dev/null
@@ -1 +0,0 @@
-export const old = true;
`);

    expect(result[0]).toMatchObject({ path: "src/old.ts", status: "D" });
  });

  it("splits a multi-file patch into viewer items", () => {
    const result = parseRemoteDiff(`From an upstream response

diff --git a/a.txt b/a.txt
index 1111111..2222222 100644
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
diff --git a/b.txt b/b.txt
new file mode 100644
--- /dev/null
+++ b/b.txt
@@ -0,0 +1 @@
+second
`);

    expect(result.map(({ path, status }) => ({ path, status }))).toEqual([
      { path: "a.txt", status: "M" },
      { path: "b.txt", status: "A" },
    ]);
    expect(result[0]?.diff).not.toContain("b/b.txt");
    expect(result[1]?.diff).toContain("+second");
  });

  it("adapts a basic rename to its destination path", () => {
    const result = parseRemoteDiff(`diff --git a/src/before.ts b/src/after.ts
similarity index 100%
rename from src/before.ts
rename to src/after.ts
`);

    expect(result[0]).toMatchObject({ path: "src/after.ts", status: "R" });
    expect(result[0]?.diff).toContain("rename from src/before.ts");
  });
});
