# Initial Processing
Building on the strengths of existing XAI solutions while addressing their limitations, this project proposes a novel approach to making CNNs more interpretable. The core idea is to leverage a combination of graph-based methods and hierarchical clustering techniques to create a more transparent and structured understanding of model behavior.

The approach begins by representing the relationships between different layers of the CNN as a graph, where nodes represent layers or features, and edges represent connections or dependencies. This graph-based representation allows for the application of established algorithms from graph theory. The algorithm generates three alternative hierarchies:

●	Union-Find on an Undirected Graph: Implements a union-find algorithm with pruning based on probability-weighted edges, designed to streamline the connectivity analysis.

●	Hierarchical Clustering: Executes hierarchical clustering on an undirected graph, utilizing complementary probability-based weights and the shortest-path metric to define clusters.

●	Strongly-Connected Components: Constructs a directed graph, applying pruning techniques based on probability-weighted edges to identify and analyze strongly connected components.
