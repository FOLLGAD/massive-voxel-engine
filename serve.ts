import { file } from "bun";
import path from "node:path";

const projectRoot = import.meta.dir;
const distDir = path.resolve(projectRoot, "public");
const indexHtmlPath = path.resolve(projectRoot, "public", "index.html");

console.log(`Serving files from: ${projectRoot}`);
console.log(`Dist directory: ${distDir}`);
console.log(`Index HTML: ${indexHtmlPath}`);

const server = Bun.serve({
  port: 5555,
  async fetch(req) {
    const url = new URL(req.url);
    const pathname = url.pathname;

    console.log(`Request received for: ${pathname}`);

    // Serve index.html for the root path
    if (pathname === "/" || pathname === "/index.html") {
      console.log("Serving index.html");
      try {
        const indexFile = file(indexHtmlPath);
        if (await indexFile.exists()) {
          return new Response(indexFile, {
            headers: {
              "Content-Type": "text/html; charset=utf-8",
              "Cross-Origin-Embedder-Policy": "require-corp",
              "Cross-Origin-Opener-Policy": "same-origin",
            },
          });
        }
        console.error(`index.html not found at ${indexHtmlPath}`);
        return new Response("Not Found", {
          status: 404,
          headers: {
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
          },
        });
      } catch (error) {
        console.error(`Error serving index.html: ${error}`);
        return new Response("Internal Server Error", {
          status: 500,
          headers: {
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
          },
        });
      }
    }

    if (pathname === "/noise-viewer") {
      console.log("Serving noise-viewer.html");
      const indexFile = file(path.resolve(projectRoot, "noise-viewer.html"));
      if (await indexFile.exists()) {
        return new Response(indexFile, {
          headers: {
            "Content-Type": "text/html; charset=utf-8",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
          },
        });
      }
    }

    const requestedPath = path.join(distDir, pathname);
    // Basic security check to prevent path traversal
    if (!requestedPath.startsWith(distDir)) {
      console.warn(`Attempted path traversal: ${pathname}`);
      return new Response("Forbidden", { status: 403 });
    }

    try {
      const jsFile = file(requestedPath);
      if (await jsFile.exists()) {
        return new Response(jsFile, {
          headers: {
            "Content-Type": "application/javascript; charset=utf-8",

            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
          },
        });
      }
      console.log(`JS file not found: ${requestedPath}`);
    } catch (error) {
      console.error(`Error serving JS file ${requestedPath}: ${error}`);
      return new Response("Internal Server Error", { status: 500 });
    }

    // Fallback to 404
    console.log(`No matching route for ${pathname}, returning 404`);
    return new Response("Not Found", {
      status: 404,
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      },
    });
  },
  error(error) {
    console.error(`Server error: ${error}`);
    return new Response("Internal Server Error", { status: 500 });
  },
});

console.log(`Listening on http://localhost:${server.port} ...`);
