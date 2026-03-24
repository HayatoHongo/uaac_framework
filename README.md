# UAAC framework implementation

This is an implementation for each procedures in UAAC framework.

In the development of the scripts in this repository, iterative consultation with ChatGPT (versions 4o, 5, 5 mini, 5.1, and 5.2; OpenAI) and Gemini (versions 2.5 Pro and 3 Pro; Google) was utilized to assist in generating the Python code. All AI-generated code was thoroughly reviewed, verified, and refined by the lead author. The lead author takes full responsibility for the integrity, accuracy, and intellectual property compliance of all code.

## Framework procedures

1. Aerial imagery acquisition
   - Generate georeferenced orthomosaic.
2. Aerial imagery clustering
   - K-means (`clustering/kmeans.py`)
   - K-means with spatial division (`clustering/kmeans.py`+`clustering/spatial_division.py`)
   - Evaluation of clustering results (`clustering/kmeans.py`+`clustering/eval_clustering.py`, reference distribution is required)
3. Sample point assignment
   - Random placement
     - Using `arcpy.management.CreateRandomPoints` in ArcGIS Pro (Esri)
       |Target density|`number_of_points_or_field`|
       |:-|-:|
       |Low |50|
       |Moderate|100|
       |High|200|
       |Very high|400|
   - Boundary-distant placement (`sample_points/boundary_distant.py`)
4. Underwater image acquisition
   - Simulated sampling from underwater orthomosaic(`sample_points/sample_uw_images.py`)
5. Underwater image analysis
   - Preprocess underwater images (`uw_image_analysis/preprocess_uw_images.py`)
   - Machine learning-based semantic segmentation model (`uw_image_analysis/model/`)
   - Calculate benthic-class proportion vectors from segmented underwater patch images (`uw_image_analysis/calc_label_proportion.py`)
6. Integration of aerial and underwater observation results
   - Calculate the representative benthic-class composition for all clusters and overall coverage (`integration_accuracy/proportion_by_cluster.py`)
7. Accuracy assessment
   - Cluster-specific proportion error $\text{Error}_c$ (`integration_accuracy/calc_cluster_proportion_errors.py`)
   - Overall coverage error $\text{Error}_{\text{overall-coverage}}$ (`integration_accuracy/calc_coverage_errors.py`)
