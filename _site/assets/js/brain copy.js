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
        
        // Load volumes.
        viewer.loadVolumes({
        volumes: [
            {
            type: "nifti1",
            header_url: "https://brainbrowser.cbrain.mcgill.ca/models/ibis_411025_living_phantom_UNC_SD_HOS_20100112_t1w_004.mnc.header",
            raw_data_url: "https://brainbrowser.cbrain.mcgill.ca/models/ibis_411025_living_phantom_UNC_SD_HOS_20100112_t1w_004.mnc.raw",
            template: {
                element_id: "volume-ui-template",
                viewer_insert_class: "volume-viewer-display"
            }
            },
            {
            type: "minc",
            header_file: document.getElementById("https://brainbrowser.cbrain.mcgill.ca/models/ibis_411025_living_phantom_UNC_SD_HOS_20100112_t1w_004.mnc.header"),
            raw_data_file: document.getElementById("https://brainbrowser.cbrain.mcgill.ca/models/ibis_411025_living_phantom_UNC_SD_HOS_20100112_t1w_004.mnc.raw"),
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
