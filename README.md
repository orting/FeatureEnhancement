FeatureEnhancement
==================

An implementation of a feature enhancement filter for preprocessing 3D images before segmentation

It is based on the two papers:
@article{ Rikxoort08,
       author = "van Rikxoort, Eva M. and van Ginneken, Bram and Klik, M. A. J. and Prokop, Mathias" ,
       title = "Supervised Enhancement Filters: Application to Fissure Detection in Chest CT Scans",
       year = "2008",
       journal = "IEEE TRANSACTIONS ON MEDICAL IMAGING",
       volume = "27",
       number = "1",
       month = "January"
}

@article{ Lassen10,
       author = "Lassen, Bianca and Kuhnigk, Jan-Martin and Friman, Ola and Krass, Stefan and  Peitgen, Heinz-Otto",
       title = "AUTOMATIC SEGMENTATION OF LUNG LOBES IN CT IMAGES BASED ON FISSURES, VESSELS, AND BRONCHI",
       year = "2010",
       journal = "ISBI"
}

It uses
FFTW - http://www.fftw.org/
FLANN - http://www.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN
CImg - http://cimg.sourceforge.net/

FFTW is GPL v2, FLANN is BSD and CImg is CeCILL/CeCILL-C. So by role of dice this code is... GPL v2