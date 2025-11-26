README

-Instacart Loader: This object has functions that build and load the csr files for efficiently using the datasets. See demo.py for examples and instructions about how to use the loader.

-demo.py: simple script demonstrating the functionality of loader

-Parquet files: the directory containing sparse matrices saved as parquet files. These files can be downloaded directly here: 

https://northeastern-my.sharepoint.com/:f:/g/personal/arnold_e_northeastern_edu/EvwsVyEtWQtAoTQvvT4qvfcBbjo9DSYGblrnR85QxShIiA?e=Fg7kKO&xsdata=MDV8MDJ8fGZlYzAzZDQ5MjZkNzRhNjdlNTBjMDhkZTJhYjdiYjY5fGE4ZWVjMjgxYWFhMzRkYWVhYzliOWEzOThiOTIxNWU3fDB8MHw2Mzg5OTUxNjU3NDQ4NTAxODh8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDJOb1lYUnpMekU1T2poaU1UWXpaRGczTUdFM01EUmpPRFpoTlRNMU1tVTRNMlppTWpJd05UTXdRSFJvY21WaFpDNTJNaTl0WlhOellXZGxjeTh4TnpZek9URTVOemN6TXpBd3xhZGMzZjhhNzExMjg0ZmYxNzA0NzA4ZGUyYWI3YmI2OXw2OGI2YjVmM2MxNGY0OWNiYjVmZWU2N2YzM2I4NDAyMA%3D%3D&sdata=YkM0RTVxUDVpeDdUbGRRMXZoS1BSLy9zREljUEtVblJqMWFmMG5wVit6Zz0%3D&ovuser=a8eec281-aaa3-4dae-ac9b-9a398b9215e7%2Carnold.e%40northeastern.edu

-instacart_data: this directory can be filled with the instacart csvs if you want to use the program to build the parquet files. 

-pset2 experiments: This folder contains all scripts used to conduct experiments for group problem set 2. Experiments include:

  -Zach's centroid clustering and visualization script
  
  -"Algorithm Tester" the code used to test all the algorithms on basket data from the groceries and instacart datasets
  
  -"KMeans basket characterizer" script that makes bar plots of aggregated items within clusters
  
  -"KMeans basket separator" a script similar to the previous that makes a bar plot of the maximum item differences in the most different clusters (weighted by size)
  
  -"KMeans entropy characterizer" opptimization script that uses various values of d and k to find the cluster entropy for the aisles the products come from
  
  -"reduction visualizer" script that visualizes dimensionally-reduced datasets using both umap and tsne to get a sense of overall structure
  
  -"SVD visualizer" script that specifically visualizes SVD dimensions using numerous plots. Intended to diagnose why KMeans was so effective
