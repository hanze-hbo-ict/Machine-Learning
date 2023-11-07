# Deel 2 - opgaveset 1

## Inleiding

We gaan deze week aan de slag met binaire classificatie: behoort een element wel (1) of niet (0) tot een verzameling? We werken met de standaard Iris-dataset van sklearn. We gaan proberen te voorspellen of bloemen wel of niet van de ondersoort Iris Virginica zijn. Hiervoor gebruiken we logistische regressie, de bijbehorende kostenfunctie en de stapsgewijze aanpassing daarvan middels _gradient descent_.

## Stappenplan

* Download de dataset m.b.v. ```load_iris()``` uit ```sklearn.datasets```.
* Vul je featurematrix X op basis van de _data_.
* De uitkomstvector y ga je vullen op basis van _target_. Standaard bevat deze array de waardes 0, 1 en 2 (resp. 'setosa', 'versicolor', 'virginica'). Maak deze binair door 0 en 1 allebei 0 te maken (niet-virginica) en van elke 2 een 1 te maken (wel-virginica). Denk erom dat y het juiste datatype en de juiste _shape_ krijgt.
* Definieer een functie ```sigmoid()``` die de sigmoïde-functie implementeert.
* Initialiseer een vector theta met 1.0'en in de juiste shape.
* Nu kun je beginnen aan de loop waarin je in **1500** iteraties:
       * De voorspellingen (denk aan sigmoid!) en de errors berekent.
       * De gradient berekent en theta aanpast. Werk in eerste instantie met een _learning rate_ van 0.01.
       * De kosten berekent.
* Als het goed is, zie je de kosten (vanaf een beginwaarde rond 8) steeds dalen kom je aan het einde rond 0,24 uit. Werk je met de niet-negatieve versie van de kostenfunctie, dan ga ja van ongeveer -8 naar -0,24.
* Experimenteer eens met andere waardes van de _learning rate_ (1.0 < alpha < 0.0) en het aantal iteraties.

Tijdens het inlevermoment demonstreer je je notebook, licht je een en ander toe en beantwoord je vragen hierover.