# Welkom bij Machine Learning

Hier vindt u onder andere de stof (per week) en de opgaven (per blok, of "deel", van twee weken).

## Inleiding

Machine learning is het onderdeel van de informatica dat zich bezighoudt met het analyseren van grote hoeveelheden data en op basis daarvan structuren herkennen of voorspellingen doen. Hoewel de basis voor dit vakgebied al in de jaren zestig van de vorige eeuw is gelegd (bijvoorbeeld door het baanbrekende werk van [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), is het pas de voorbije twee decennia echt tot grote bloei gekomen. Dit heeft vooral te maken met de enorme hoeveelheid data die heden ten dage beschikbaar wordt gesteld (een fenomeen bekend onder de term [information explosion](https://en.wikipedia.org/wiki/Information_explosion) in combinatie met de grote en goedkope computationele kracht van hedendaagse computers en data-centra.

## Opgaven

Op deze pagina's zijn de weekopgaven van dit onderdeel te vinden (zie het menu links).

De practicumopdrachten worden bij voorkeur gemaakt in **duo's**. Tijdens het eerste practicum wordt klassikaal een begin gemaakt met de opgaven: er wordt uitgelegd hoe te werken met [NumPy](http://www.numpy.org/) en hoe je lineaire algebra gebruikt in het domein van ML. Ook worden hier de weekopgaven verder toegelicht.

Tijdens het tweede, vierde, zesde en achtste practicum worden (tijdens de geroosterde oplevermomenten) afspraken gemaakt met de duo's om het werk tot dan toe te bespreken en te beoordelen.

Er zijn vier sets weekopgaven die uiteindelijk allemaal in de voorlaatste week van het blok moeten zijn afgerond. Elke set moet met een **voldoende** cijfer beoordeeld zijn. Eventueel **reparatiewerk** wordt tijdens de oplevermomenten besproken. Mocht dat niet voor de deadline zijn afgerond, dan is er een uitloopmogelijkheid in de laatste week van het blok. Dat geldt dan wel als een herkansing. Een niet-ingeleverde opdracht krijgt het cijfer 1.

## Opstarten

De opgaven voor elke week gaan uit van een zipje met startcode. Het geheel gaat uit van een aantal dependencies. Deze dependencies hebben we voor het gemak in een [`requirements.txt`](files/requirements.txt){: download="requirements.txt"} gezet. Je kunt het beste een virtuele omgeving aanmaken en hierin met pip in één keer alle dependencies installeren. Hier een voorbeeld voor MacOS:

```
baba@aurelia ~ % python -m venv ml
baba@aurelia ~ % cd ml 
baba@aurelia ml % cp ~/Downloads/requirements.txt .
baba@aurelia ml % source bin/activate
(ml) baba@aurelia ml % python -m pip install -r requirements.txt 
...
(ml) baba@aurelia ml % 
```

en voor Windows (Terminal/PowerShell)
```
PS C:\Users\ez> python -m venv ml
PS C:\Users\ez> cd ml
PS C:\Users\ez\ml> cp ~/Downloads/requirements.txt .
PS C:\Users\ez\ml> Scripts/activate
(ml) PS C:\Users\ez\ml> python -m pip install -r requirements.txt
...
(ml) PS C:\Users\ez\ml> 
```

## Globale planning

| Weeknummer | Onderwerp | Inleveren |
| ---------- | --------- | --------- |
| 1	| Tools en technieken | |	 
| 2	| Lineaire regressie | opgaven deel 1 |
| 3	| Logistische regressie | |
| 4	| Neurale netwerken | opgaven deel 2 |
| 5	| Model-evaluatie en Hyperparameter tuning | |
| 6	| Andere modellen en Ensemble learning | opgaven deel 3 |
| 7	| Dimensionaliteitsreductie en (grote) taalmodellen | |
| 8	| Recurrente neural netwerken | opgaven deel 4 |

## Stof en opgaven per week

### Deel 1

* [Week 1](deel1/week1.md)
* [Week 2](deel1/week2.md)
* [Inlevermoment](deel1/inleveren.md)

### Deel 2

* [Week 3](deel2/week3.md)
* [Week 4](deel2/week4.md)
* [Inlevermoment](deel2/inleveren.md)

### Deel 3

* [Week 5](deel3/week5.md)
* [Week 6](deel3/week6.md)
* [Inlevermoment](deel3/inleveren.md)

### Deel 4

* [Week 7](deel4/week7.md)
* [Week 8](deel4/week8.md)
* [Inlevermoment](deel4/inleveren.md)
