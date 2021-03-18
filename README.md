# AtlasAutoEncoder
The AutoEncoder compress the four-momentum of a sample of simulated particles from 4 to 3 variables.

Here, you can find the codes for implementation of autoencoder-decoder network to compress the data from the ATLAS experiments from 4 to 3 variables, and reconstruct the same.

## Data

The ATLAS experiments being conducted at the Large Hadron Collider at CERN produce millions of proton-proton collision events. Here, I analyse the variables from the 4-momentum from the leading jets.
- [Training Set](https://github.com/swaingotnochill/AtlasAutoEncoder/blob/main/Processed%20Data/train%20.csv)
- [Test Set](https://github.com/swaingotnochill/AtlasAutoEncoder/blob/main/Processed%20Data/test.csv)

### The 4 Variables:
- *pt*: transverse momentum pT 
- *eta*: pseudorapidity η 
- *phi*: azimuthal angle φ 
- *eta*: energy E

## Environment
Python 3.5    

## Dependencies  
- numpy      
- pandas 
- matplotlib 
- sys 
- torch  
- fastai == 1.0.61
- scipy 
- seaborn
- corner 

# Code
The Colab Notebook consist of the following modules:
1. Loading the dataset.  
The analysis has been done by preprocessing the data in two ways and comparing the results by auto-ecoding the two sets of preprocessed data. Preprocessing of the data:
- Normalisation of data: Subtract the mean of the training set and divide by standard deviation of the training set. 
- Singlular Value Decomposition(SVD): Mathematical technique used for dimensionality reduction. We will add the singular values to original dataframe and then pass it through the deep autoencoder network.  
2. Histogram to visulaise the data distribution. 
3. Train the Model. 
(NOTE: You can just use the pretrained model by loading it) 
-[Model](https://github.com/swaingotnochill/AtlasAutoEncoder/blob/main/models/AE_GivenNetworkWithSVD_v2.pth)
5. Reconstruction of the data using the auto encoder-decoder model.  
The plots for both the reconstructed normalised as well as custom standardised data are shown.  
5. Reconstruction Loss (Residual).  
`Residual = (Predicted Data - Original Data) / Original Data`  
## Usage

##### 1. Install dependencies
##### 2. Downloading the dataset.
Download the datasets and store them in a folder `processed_data`:
##### 2. Run the Jupyter notebook.
The various plots can be visualised in the Jupyter Notebook

# Conclusions

##### 1. Validation Loss : 0.000028
##### 2. Residual : 2.9843 e-05
