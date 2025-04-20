import { mkSimplexNoise, type SimplexNoise, type PRNG } from "../src/noise";

// Simple deterministic PRNG based on seed
// From https://stackoverflow.com/a/19303725/1480448
function mulberry32(a: number): PRNG {
  let aa = a;
  return () => {
    // biome-ignore lint/suspicious/noAssignInExpressions: <explanation>
    let t = (aa += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const canvas = document.getElementById("noiseCanvas") as HTMLCanvasElement;
// biome-ignore lint/style/noNonNullAssertion: <explanation>
const ctx = canvas.getContext("2d")!;
const scaleInput = document.getElementById("scale") as HTMLInputElement;
const seedInput = document.getElementById("seed") as HTMLInputElement;
const redrawButton = document.getElementById(
  "redrawButton"
) as HTMLButtonElement;

let noise: SimplexNoise;
let currentScale: number;
let currentSeed: number;

function drawNoise() {
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  // Generate noise for each pixel
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Map canvas coordinates to noise coordinates using scale
      const noiseX = x * currentScale;
      const noiseY = y * currentScale;

      // Get noise value in range [-1, 1]
      const value = noise.noise2D(noiseX, noiseY);

      // Map noise value to grayscale [0, 255]
      const color = Math.floor((value + 1) * 0.5 * 255);

      const index = (y * width + x) * 4;
      data[index] = color; // R
      data[index + 1] = color; // G
      data[index + 2] = color; // B
      data[index + 3] = 255; // A (fully opaque)
    }
  }

  ctx.putImageData(imageData, 0, 0);
  console.log(`Noise drawn with scale: ${currentScale}, seed: ${currentSeed}`);
}

function setupAndDraw() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  currentScale = Number.parseFloat(scaleInput.value);
  const seedValue = Number.parseInt(seedInput.value, 10);
  currentSeed = Number.isNaN(seedValue) ? Date.now() : seedValue; // Use timestamp if seed is invalid
  seedInput.value = currentSeed.toString(); // Update input if timestamp was used

  console.log(`Initializing noise with seed: ${currentSeed}`);
  const prng = mulberry32(currentSeed);
  noise = mkSimplexNoise(prng);

  drawNoise();
}

// Event listeners
redrawButton.addEventListener("click", setupAndDraw);
scaleInput.addEventListener("input", () => {
  currentScale = Number.parseFloat(scaleInput.value);
  drawNoise(); // Redraw immediately on scale change
});
window.addEventListener("resize", setupAndDraw);

// Initial setup
setupAndDraw();
