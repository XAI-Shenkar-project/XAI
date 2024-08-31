# XAI GUI
To make the results of the graph-based analysis and hierarchical clustering accessible to users, a user-friendly interface has been developed. This interface allows users to explore the connections and clusters interactively, providing visual explanations that can be easily understood and interpreted. 

The user interface (UI) was developed using pyQT, a set of Python bindings for the Qt application framework. This choice was driven by pyQT's ability to create highly interactive and visually appealing interfaces. Its flexibility in designing custom widgets allowed us to tailor the UI to meet the needs of AI researchers and data scientists, making it the preferred tool for this project.

The external representation also includes the design and layout of the user interface (UI), where users interact with the system. The UI provides a dashboard for inputting data, selecting models, and viewing results. It includes graphical elements like charts, graphs, and interactive plots that visualize the model's behavior and explanations:

●	UI Files: Contains all the necessary user interface components, ensuring a seamless interaction between the user and the XAI system.

●	Dendrogram Images: Includes visual representations of the dendrograms generated for each CNN model and graph type, providing insights into the hierarchical relationships within the data.

●	main.py: The central Python script responsible for executing the system and defining its core functionality, integrating the various processing components with the user interface.

●	user_common_group.txt: A text file that records all user-defined naming conventions, ensuring consistency and traceability throughout the analysis process.
