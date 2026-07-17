const renderMermaid = () => {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
  });

  mermaid.run({
    querySelector: "pre.mermaid",
  });
};

if (typeof document$ !== "undefined") {
  document$.subscribe(renderMermaid);
} else {
  window.addEventListener("load", renderMermaid);
}
