# Přehled úkolů

## Úkol 1: Perceptron pro binární klasifikaci

Implementace jednoduchého perceptronu pro klasifikaci bodů v rovině podle jejich pozice vůči přímce. Perceptron se učí
pomocí iterativního algoritmu rozlišovat body nad a pod danou přímkou, přičemž výsledná rozhodovací hranice by měla
aproximovat původní přímku. Vizualizace výsledků umožňuje ověřit správnost implementace a demonstrovat schopnost
natrénovaného perceptronu klasifikovat data.

## Úkol 2: Neuronová síť pro XOR problém

Vytvoření neuronové sítě s jednou skrytou vrstvou pro řešení problému XOR, který není lineárně separabilní. Neuronová
síť používá sigmoidní aktivační funkci a algoritmus zpětné propagace pro učení, přičemž cílem je správně klasifikovat
čtyři kombinace vstupů XOR operace (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. Výsledná vizualizace ukazuje nelineární
rozhodovací hranici naučenou sítí, což demonstruje schopnost neuronových sítí modelovat komplexní vztahy mezi vstupy a
výstupy.

## Úkol 3: Hopfieldova síť pro rozpoznávání vzorů

Implementace Hopfieldovy sítě pro ukládání a rekonstrukci vzorů, jako jsou číslice nebo písmena. Síť používá asociativní
paměť založenou na váhové matici, která se učí z předložených vzorů a následně dokáže opravit poškozené nebo neúplné
vzory na jejich původní formu. Rekonstrukce může probíhat synchronně (všechny neurony najednou) nebo asynchronně (
postupná aktualizace jednotlivých neuronů), přičemž obě metody vedou k stabilnímu stavu odpovídajícímu některému z
naučených vzorů.

## Úkol 4: Q-learning pro navigaci agenta v herním prostředí

Vytvoření agenta používajícího Q-learning algoritmus pro navigaci v herním prostředí reprezentovaném maticí, kde
jednotlivé buňky mohou obsahovat odměny nebo překážky. Agent se postupně učí optimální strategii pro dosažení cíle (
sýrů) a vyhýbání se nežádoucím polím (zdi, kočky) pomocí aktualizace Q-hodnot. Vizualizace ukazuje, jak agent postupně
nachází optimální cestu k cíli, a to i v prostředí s více cíly.

## Úkol 5: Balancování tyče pomocí Deep Q-Learning

Implementace řešení klasického problému "Cart-Pole" (balancování tyče) pomocí techniky Deep Q-Learning. Agent se učí
udržet tyč ve vzpřímené poloze na vozíku pomocí neuronové sítě, která aproximuje Q-hodnoty pro jednotlivé akce. Trénink
probíhá v simulovaném prostředí, kde agent dostává odměny za každý krok, kdy je tyč stále vzpřímená, a výsledné řešení
je vizualizováno pomocí grafu učení a animace demonstrující úspěšné balancování.

## Úkol 6: Fraktály pomocí L-systémů

Vytvoření aplikace pro generování a vizualizaci fraktálů pomocí L-systémů, což jsou formální gramatiky používané pro
modelování růstu rostlin a generování komplexních grafických struktur. Aplikace umožňuje definovat axiom (počáteční
řetězec), pravidla přepisování a parametry vykreslování jako úhel otáčení nebo délku čáry. Vykreslené fraktály mohou být
interaktivně upravovány a ukládány, přičemž aplikace podporuje různé styly vykreslování včetně nastavení barvy a
tloušťky čar.

## Úkol 7: Fraktálový Generátor IFS

Implementace generátoru fraktálů založeného na Iterovaných Funkčních Systémech (IFS). Program využívá sadu
transformačních funkcí, které definují lineární transformace v 3D prostoru, a algoritmus náhodné iterace pro generování
bodů fraktálu. Výsledné vizualizace jsou zobrazeny jako interaktivní 3D grafy, které umožňují rotaci a přiblížení
fraktálních struktur. Různé sady transformací umožňují vytvářet rozmanité fraktální vzory v trojrozměrném prostoru.

## Úkol 8: Generátor Fraktálů Mandelbrotovy a Juliovy Množiny

Vytvoření generátoru pro vizualizaci dvou známých komplexních fraktálů - Mandelbrotovy a Juliovy množiny. Program
implementuje iterační algoritmy pro výpočet příslušnosti bodů do těchto množin, generuje statické snímky a animace
zoomování do zajímavých oblastí fraktálů. Výsledné vizualizace zahrnují barevné mapování, které zvýrazňuje struktury
fraktálů, a interaktivní animace ukazující nekonečnou složitost Mandelbrotovy množiny a Juliovy množiny s různými
komplexními konstantami.

## Úkol 9: Generátor Fraktální Krajiny

Implementace generátoru fraktální krajiny pomocí metody půlení úseček s náhodným posunutím (midpoint displacement).
Algoritmus začíná s přímkou a postupně ji dělí, přičemž středové body jsou náhodně posunuty, což vytváří realistický
profil krajiny. Aplikace umožňuje nastavení parametrů jako počet iterací, výchylka a drsnost, které ovlivňují výsledný
vizuální vzhled, a podporuje generování vícevrstvých krajin pro vytváření komplexnějších scén.

## Úkol 10: Neuronová síť pro predikci bifurkačního diagramu

Implementace neuronové sítě, která se učí predikovat bifurkační diagram logistického zobrazení - nelineárního
dynamického systému, který vykazuje deterministické i chaotické chování. Síť je natrénována na datech z různých hodnot
řídícího parametru a dokáže aproximovat komplexní struktury bifurkačního diagramu. Analýza výsledků ukazuje, že
neuronová síť je relativně přesná v oblastech s jednodušším chováním, ale má omezenou schopnost modelovat detaily v
chaotických režimech.

## Úkol 11: Chaotický pohyb dvojitého kyvadla (NENÍ IMPLEMENTOVÁNO)

Implementace fyzikální simulace dvojitého kyvadla, které je klasickým příkladem chaotického systému. Projekt zahrnuje
matematický model vycházející z Lagrangiánu a Euler-Lagrangeových rovnic pohybu, numerické řešení diferenciálních rovnic
pomocí funkce odeint a vizualizaci výsledného chaotického pohybu. Simulace demonstruje základní vlastnosti chaotických
systémů, jako je extrémní citlivost na počáteční podmínky, kdy nepatrná změna počátečního stavu vede k zásadně odlišnému
dlouhodobému chování systému, přestože je plně deterministický.

## Úkol 12: Celulární automat lesního požáru

Implementace dvourozměrného celulárního automatu simulujícího šíření lesního požáru v dynamickém prostředí. Simulace
zahrnuje čtyři možné stavy buněk (prázdné místo, strom, hořící strom, spáleniště) a pravidla přechodu mezi nimi.
Experiment zkoumá vliv různých typů okolí (Von Neumannovo, Moorovo, hexagonální) na vzory šíření požáru. Vizualizace
ukazuje emergentní chování systému, včetně dynamické rovnováhy mezi růstem nových stromů a jejich ničením požárem.
