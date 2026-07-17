/**
 * Find approximate location of ES module syntax errors by scanning with a naive `{`/`}` delta,
 * appending closing braces so truncated files remain bracket-balanced (strings/comments not handled).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const filePath = path.join(root, "frontend", "scripts", "app", "app.js");
const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);

async function parse(src) {
  try {
    await import(`data:text/javascript;charset=utf-8,${encodeURIComponent(src)}`);
    return { ok: true };
  } catch (e) {
    return { ok: false, msg: e.message };
  }
}

function braceDelta(line) {
  let n = 0;
  for (const ch of line) {
    if (ch === "{") n++;
    else if (ch === "}") n--;
  }
  return n;
}

async function main() {
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    const partial = lines.slice(0, i + 1).join("\n");
    depth += braceDelta(lines[i]);
    const closer = depth > 0 ? `\n${"})]}".slice(0, 0)}${""}` : "";

    const fixed =
      depth > 0 ? `${partial}\n${"}\n".repeat(depth)}` : partial;

    const r = await parse(fixed);
    if (!r.ok) {
      console.log("First line index (1-based) where balanced-prefix parse fails:", i + 1);
      console.log("Message:", r.msg);
      console.log("Brace depth after line (naive):", depth);
      const lo = Math.max(1, i + 1 - 8);
      const hi = Math.min(lines.length, i + 1 + 4);
      for (let j = lo; j <= hi; j++) {
        console.log(String(j).padStart(5, " ") + "|" + lines[j - 1]);
      }
      process.exit(0);
    }
  }
  console.log("No failure found with naive brace closing (unexpected)");
}

await main();
