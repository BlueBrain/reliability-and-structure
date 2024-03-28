# About

This folder contains a notebook describing a method by which to select neighbourhoods in the BlueBrian V5 excitatory-excitatory circuit (further just called "circuit"). Selecting neghbourhoods (in most cases, the number of neighbourhoods selected is 50) is necessary to determine which collections of neighbourhoods perform best in classifying activity.

## Approach

Neighbourhoods (a single vertex, its adjacent vertices, and all edges among them) are selected in several ways. The first step is to compute values (such as spectral gaps and clustering coefficients) for every neighbourhood, in particular our density parameter `rc_per_nodes`, which couts the number of pairs of reciprocal connections in a neighbourhood, divided by the number of nodes in the neighbourhood.

### Random selections

Select randomly:
1. neighbourhoods from the full circuit
2. neighbourhoods from the 1% of neighbourhoods with lowest density paramater value
3. neighbourhoods from the 1% of neighbourhoods with highest density paramater value

### Single selections

For every parameter, select
1. neighbourhoods having the highest paramater value from the full circuit
2. neighbourhoods having the lowest paramater value from the full circuit

### Double selections

Restrict to the 1% of neighbourhoods having the lowest density paramater value. The for every other parameter, select
1. neighbourhoods having the highest paramater value from that subcircuit
2. neighbourhoods having the lowest paramater value from that subcircuit
