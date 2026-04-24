// Initializes a BrainBrowser Volume Viewer on #brainBrowserWrapper.
//
// The earlier (Jekyll) version of this script worked by accident: it called
// viewer.render() + viewer.loadVolumes() synchronously before the color map
// had finished loading from McGill's CDN. On a slow load that race triggers
// a "No color map set" error deep inside getSliceImage, which in turn hits
// a BrainBrowser bug where triggerEvent mutates a string argument — Safari
// enforces strict-mode readonly on that string and throws.
//
// Fix: load the color map locally (no network race) AND defer render/load
// until the color map's callback fires.
(function () {
  function start() {
    if (typeof BrainBrowser === "undefined") {
      return setTimeout(start, 100);
    }

    BrainBrowser.VolumeViewer.start("brainBrowserWrapper", function (viewer) {
      viewer.addEventListener("volumesloaded", function () {
        console.log("BrainBrowser: volumes loaded.");
      });

      viewer.setPanelSize(256, 256);

      // Third argument is a callback invoked once the color map has loaded.
      viewer.loadDefaultColorMapFromURL(
        "/assets/brainbrowser/colormaps/gray_scale.txt",
        "#FF0000",
        function () {
          viewer.render();
          viewer.loadVolumes({
            volumes: [
              {
                type: "nifti1",
                nii_url: "/assets/brain/7T.nii",
                template: {
                  element_id: "volume-ui-template",
                  viewer_insert_class: "volume-viewer-display",
                },
              },
            ],
          });
        }
      );
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
