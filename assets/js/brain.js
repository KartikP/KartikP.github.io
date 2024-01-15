    BrainBrowser.VolumeViewer.start("brainBrowserWrapper", function(viewer) {
        
        console.log(viewer)

        // Add an event listener.
        viewer.addEventListener("volumesloaded", function() {
        console.log("Viewer is ready!");
        });
    
        // Load the default color map.
        // (Second argument is the cursor color to use).
        viewer.loadDefaultColorMapFromURL('https://brainbrowser.cbrain.mcgill.ca/color_maps/gray_scale.txt', "#FF0000");
    
        // Set the size of slice display panels.
        viewer.setPanelSize(256, 256);
    
        // Start rendering.
        viewer.render();
        console.log("Loading volumes")
        // Load volumes.
        viewer.loadVolumes({
        volumes: [
            {
            type: "nifti1",
            nii_url: "http://kartikpradeepan.com/assets/brain/7T.nii",
            template: {
                element_id: "volume-ui-template",
                viewer_insert_class: "volume-viewer-display"
            }
            }
        ],
        overlay: {
            template: {
                element_id: "volume-ui-template",
                viewer_insert_class: "volume-viewer-displayy"
            }
        }
        });
    });

