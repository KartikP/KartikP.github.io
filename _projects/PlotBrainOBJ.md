---
layout: page
title: Plot your brain MRI
description: How to plot OBJ files in Plotly
img: 
category: neuroscience
related_publications: 
url: 
date: 2024-01-15
---

[To see the results of this code click here](/blog/2024/My-Brain/)

[To convert .nii files quickly to .obj without Freesurfer processing](https://github.com/rordenlab/nii2meshWeb)

```python
import plotly.graph_objects as go
import numpy as np
```


```python
brain = "/content/drive/MyDrive/Data Science/MRI/brain.obj"
```


```python
# Function to read the .obj file
def read_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f'):
                face = [int(i.split('/')[0]) for i in line.strip().split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)
```


```python
vertices, faces = read_obj(brain)
```


```python
vertices
```




    array([[  50.666779,  -28.194384,  -21.338442],
           [  50.121082,  -27.810968,  -20.198452],
           [   3.378802, -100.455139,    4.870674],
           ...,
           [ -12.782207,   66.600349,   16.896309],
           [ -13.962158,   66.610062,   17.925493],
           [ -10.211386,   31.746548,   -1.28224 ]])




```python
faces
```




    array([[34663, 28109, 28500],
           [ 3049, 15445, 37208],
           [ 1154,     1,     2],
           ...,
           [50243, 51376, 44998],
           [70636, 50891, 69542],
           [49939, 70369, 74995]])




```python
# Create a mesh object
pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

mesh = go.Mesh3d(
    x=vertices[:,0],
    y=vertices[:,1],
    z=vertices[:,2],
    intensity=vertices[:,2],
    i=faces[:,0]-1,
    j=faces[:,1]-1,
    k=faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)
mesh.update(cmin=-7,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );

```


```python
# Create a figure and add the mesh object to it
fig = go.Figure(data=[mesh])
camera = dict(
    eye=dict(x=5, y=1.5, z=0)  # Adjust these values as needed
)
fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            title=''  # Remove the x-axis label
        ),
        yaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            title=''
        ),
        zaxis=dict(
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            title=''
        ),
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=0),  # Removes white margins around the plot
    paper_bgcolor='rgba(0,0,0,0)',  # Set the background color of the paper to transparent
    plot_bgcolor='rgba(0,0,0,0)'    # Set the background color of the plot to transparent
)
fig.show()
fig.write_html('/content/drive/MyDrive/Data Science/MRI/brain.html')
```

## Subplots containing a variety of different views of the .nii to .obj conversion.


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
```


```python
face = "/content/drive/MyDrive/Data Science/MRI/nii2mesh_sub-009_T1w.obj"
T7_raw = "/content/drive/MyDrive/Data Science/MRI/nii2mesh_7T.obj"
T7_white = "/content/drive/MyDrive/Data Science/MRI/nii2mesh_c2_KP_anatomical.obj"
T7_gray = "/content/drive/MyDrive/Data Science/MRI/nii2mesh_c1_KP_anatomical.obj"
T7_csf = "/content/drive/MyDrive/Data Science/MRI/nii2mesh_c3_KP_anatomical.obj"

# Default camera settings
camera = dict(
    eye=dict(x=3, y=1.5, z=0)  # Adjust these values as needed
)
```


```python
# Function to read the .obj file
def read_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f'):
                face = [int(i.split('/')[0]) for i in line.strip().split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)
```

## 3T Face and 7T raw


```python
face_vertices, face_faces = read_obj(face)
T7_raw_vertices, T7_raw_faces = read_obj(T7_raw)
# Create a mesh object
pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

face_mesh = go.Mesh3d(
    x=face_vertices[:,0],
    y=face_vertices[:,1],
    z=face_vertices[:,2],
    intensity=face_vertices[:,2],
    i=face_faces[:,0]-1,
    j=face_faces[:,1]-1,
    k=face_faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)
T7_raw_mesh = go.Mesh3d(
    x=T7_raw_vertices[:,0],
    y=T7_raw_vertices[:,1],
    z=T7_raw_vertices[:,2],
    intensity=T7_raw_vertices[:,2],
    i=T7_raw_faces[:,0]-1,
    j=T7_raw_faces[:,1]-1,
    k=T7_raw_faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)

face_mesh.update(cmin=-3.3109,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );
T7_raw_mesh.update(cmin=-3.31909,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );
```


```python
# Create a figure and add the mesh object to it
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
    horizontal_spacing=0.02  # Adjust the spacing as needed
)

fig.add_trace(face_mesh, row=1, col=1)
fig.add_trace(T7_raw_mesh, row=1, col=2)


axis_settings = dict(
    showbackground=False,
    showgrid=False,
    zeroline=False,
    showticklabels=False,
    showline=False,
    title=""
)

# Add titles as 3D text
fig.add_trace(go.Scatter3d(
    x=[np.mean(face_vertices[:,0])],
    y=[np.mean(face_vertices[:,1])],
    z=[np.max(face_vertices[:,2]) * 1.2],  # Position above the highest point of the mesh
    text=["Face (from separate 3T)"],
    mode="text",
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=[np.mean(T7_raw_vertices[:,0])],
    y=[np.mean(T7_raw_vertices[:,1])],
    z=[np.max(T7_raw_vertices[:,2]) * 1.2],  # Position above the highest point of the mesh
    text=["Skulled stripped (7T)"],
    mode="text",
), row=1, col=2)

fig.update_layout(
    scene=dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        camera=camera
    ),
    scene2=dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)

fig.show()
fig.write_html('/content/drive/MyDrive/Data Science/MRI/face_raw.html')
```

## 7T white and gray matter, and CSF


```python
white_vertices, white_faces = read_obj(T7_white)
gray_vertices, gray_faces = read_obj(T7_gray)
csf_vertices, csf_faces = read_obj(T7_csf)

# Create a mesh object
pl_mygrey=[0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

white_mesh = go.Mesh3d(
    x=white_vertices[:,0],
    y=white_vertices[:,1],
    z=white_vertices[:,2],
    intensity=white_vertices[:,2],
    i=white_faces[:,0]-1,
    j=white_faces[:,1]-1,
    k=white_faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)
gray_mesh = go.Mesh3d(
    x=gray_vertices[:,0],
    y=gray_vertices[:,1],
    z=gray_vertices[:,2],
    intensity=gray_vertices[:,2],
    i=gray_faces[:,0]-1,
    j=gray_faces[:,1]-1,
    k=gray_faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)
csf_mesh = go.Mesh3d(
    x=csf_vertices[:,0],
    y=csf_vertices[:,1],
    z=csf_vertices[:,2],
    intensity=csf_vertices[:,2],
    i=csf_faces[:,0]-1,
    j=csf_faces[:,1]-1,
    k=csf_faces[:,2]-1,
    colorscale=pl_mygrey,
    showscale=False
)

white_mesh.update(cmin=-3.3109,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );
gray_mesh.update(cmin=-3.31909,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );
csf_mesh.update(cmin=-3.31909,
               lighting=dict(ambient=0.18,
                             diffuse=1,
                             fresnel=0.1,
                             specular=1,
                             roughness=0.05,
                             facenormalsepsilon=1e-15,
                             vertexnormalsepsilon=1e-15),
               lightposition=dict(x=100,
                                  y=200,
                                  z=0
                                 )
                      );
```


```python
# Create a figure and add the mesh object to it
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}, {'type': 'mesh3d'}]],
    horizontal_spacing=0.02  # Adjust the spacing as needed
)

fig.add_trace(white_mesh, row=1, col=1)
fig.add_trace(gray_mesh, row=1, col=2)
fig.add_trace(csf_mesh, row=1, col=3)


axis_settings = dict(
    showbackground=False,
    showgrid=False,
    zeroline=False,
    showticklabels=False,
    showline=False,
    title=""
)

# Add titles as 3D text
fig.add_trace(go.Scatter3d(
    x=[np.mean(white_vertices[:,0])],
    y=[np.mean(white_vertices[:,1])],
    z=[np.max(white_vertices[:,2]) * 1.2],  # Position above the highest point of the mesh
    text=["White matter (7T)"],
    mode="text",
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=[np.mean(gray_vertices[:,0])],
    y=[np.mean(gray_vertices[:,1])],
    z=[np.max(gray_vertices[:,2]) * 1.2],  # Position above the highest point of the mesh
    text=["Gray matter (7T)"],
    mode="text",
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=[np.mean(csf_vertices[:,0])],
    y=[np.mean(csf_vertices[:,1])],
    z=[np.max(csf_vertices[:,2]) * 1.2],  # Position above the highest point of the mesh
    text=["CSF (7T)"],
    mode="text",
), row=1, col=3)

fig.update_layout(
    scene=dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        camera=camera
    ),
    scene2=dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        camera=camera
    ),
      scene3=dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)

fig.show()
fig.write_html('/content/drive/MyDrive/Data Science/MRI/white_gray_csf.html')
```
