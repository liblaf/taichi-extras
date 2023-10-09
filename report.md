## Project Target

- **Data Collection:** pre- & post- surgery CT of individuals
- **Target**
  - **Input:** pre-surgery CT + surgical details (post-surgery skeleton)
  - **Output:** post-surgery face

## Last Week

- test "point cloud -> mesh" on "face before surgery"

|                                           Rigid Aligned                                           | Non-Rigid Deformation                      |
| :-----------------------------------------------------------------------------------------------: | ------------------------------------------ |
| ![Rigid Aligned](/home/liblaf/Pictures/Screenshots/Screenshot%20from%202023-09-26%2019-43-33.png) | ![Non-Rigid Deformation](./snapshot02.png) |

## TODO List

- [ ] preprocessing
  - [ ] CT -> mesh
    - [x] CT -> point cloud
    - [x] point cloud -> mesh
    - [ ] test cases
      - [x] face before surgery
      - [ ] **(next week) skull before surgery**
      - [ ] **(next week) face after surgery**
      - [ ] **(next week) skull after surgery**
  - [x] MeshFix
  - [x] TetGen
  - [ ] **(next week) (partial complete) composite**
- [x] simulation
- [ ] postprocessing
  - [ ] split
