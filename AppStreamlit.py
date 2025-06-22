import plotly.express as px
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import matplotlib
from ScheduleOptFunctions import (
    update_reload_times,
    remove_tables_by_source,
    optimize_csv_data,
    seconds_to_hhmm,
    round_time_up
)

# Streamlit Application Title
st.title("Datalake Flow Optimization")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV optimizer file", type="csv")

# Control variables for state
optimize_from_csv = False
optimize_with_updated_data = False
df = None

# If a CSV file is uploaded, read and display it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded file:")
    st.dataframe(df.head())  # Display the first few rows for preview

    # Sidebar tabs with radio buttons for functionality selection
    option = st.sidebar.radio("Select a functionality", ["Lineage Display", "Schedule Optimization", "Sankey Diagrams"])

    # **Functionality 1**
    if option == "Lineage Display":
        
        st.header("Lineage Display")

        # Check if `df` is None before accessing its columns
        if df is not None:
            # Select required columns
            df = df[['Mart', 'Mart_Domain', 'Destination', 'Destination_Domain', 'Source', 'Source_Domain', 'Reload_Time', 'Median_Load_Time','Bottleneck_Time']]
        else:
            st.error("Please upload a valid CSV file.")

        # Get the list of unique `Mart_Domain` values
        mart_domains = df['Mart_Domain'].unique()

        # Dropdown for selecting a `Mart_Domain`
        mart_domain = st.selectbox("Select a Mart Domain", mart_domains)

        # Filter available marts for the selected `Mart_Domain`
        marts = df[df['Mart_Domain'] == mart_domain]['Mart'].unique()

        # Dropdown for selecting a `Mart` within the selected `Mart_Domain`
        mart_selected = st.selectbox("Select a Mart", marts)

        # Create a directed graph
        G = nx.DiGraph()

        # Function to assign colors based on the node level
        def assign_color(domain):
            if domain == mart_domain:
                return 'red'  # Lowest level
            elif domain == 'intermediate':
                return 'yellow'  # Intermediate level
            elif domain == 'staging':
                return '#ADD8E6'  # Light blue for the highest level (staging)
            return 'gray'  # Default color for other domains

        # Add nodes and edges to the graph
        for index, row in df.iterrows():
            source_node = row['Source']
            dest_node = row['Destination']

            # Assign colors to nodes based on their domain
            source_color = assign_color(row['Source_Domain'])
            dest_color = assign_color(row['Destination_Domain'])

            # Add nodes with their color and additional information
            G.add_node(source_node, color=source_color, Mart_Domain=row['Mart_Domain'],
                    Source_Domain=row['Source_Domain'], Reload_Time=row.get('Reload_Time', 'N/A'))
            G.add_node(dest_node, color=dest_color, Mart_Domain=row['Mart_Domain'],
                    Source_Domain=row['Source_Domain'], Reload_Time=row.get('Reload_Time', 'N/A'))

            # Add the edge between source and destination
            G.add_edge(source_node, dest_node)

        # Node of interest
        node_of_interest = mart_selected

        # Find all nodes directly or indirectly dependent on the node of interest
        dependent_nodes = nx.ancestors(G, node_of_interest)
        dependent_nodes.add(node_of_interest)  # Include the node of interest in the list

        # Find all edges associated with these nodes
        edges_in_dependency_subgraph = [(u, v) for u, v in G.edges() if u in dependent_nodes and v in dependent_nodes]

        # Create the dependency subgraph
        dependency_subgraph = G.edge_subgraph(edges_in_dependency_subgraph).copy()

        # Use spring_layout to minimize edge crossings
        pos = nx.spring_layout(dependency_subgraph,
                               k=None,
                               iterations=10000,
                               seed=5,
                               )  # Increase iterations for better layout

        # Extract node colors for visualization
        node_colors = [dependency_subgraph.nodes[node]['color'] for node in dependency_subgraph.nodes]

        # Draw the subgraph with the chosen layout
        plt.figure(figsize=(20, 12))  # Increase figure size for better visualization

        total_nodes = len(dependency_subgraph.nodes)
        base_node_size = 250
        node_size = max(500, base_node_size * (50 / max(total_nodes, 1)))
        font_size = max(8, int(node_size / 250))
        
        # Patch for `np.alltrue`
        # NumPy 2.0 removed the `np.alltrue` function. Some libraries, like older versions of NetworkX,
        # might still reference it. This patch defines `np.alltrue` as an alias for `np.all` 
        # to ensure compatibility and avoid runtime errors.
        if not hasattr(np, 'alltrue'):
            np.alltrue = np.all

        # Visualize the graph with node labels
        nx.draw(dependency_subgraph, 
                pos,
                with_labels=True,
                node_size=node_size,
                arrowsize = 20,
                node_color=node_colors,
                font_size=font_size,
                font_weight='bold',
                edge_color='black', # Curved edges
                arrows=True)

        # Add labels with additional information for 'staging' nodes
        for node in dependency_subgraph.nodes:
            # Only add the 'Reload Time' label to 'staging' nodes
            if dependency_subgraph.nodes[node]['color'] == '#ADD8E6':  # 'Staging' nodes
                reload_time = dependency_subgraph.nodes[node].get('Reload_Time', 'N/A')

                # Convert Reload Time from seconds to 'HH:MM' format
                if reload_time != 'N/A':
                    reload_time = seconds_to_hhmm(reload_time)

                node_info = f"Schedule: {reload_time}"

                # Position the label slightly to the right (e.g., +0.18 in X)
                x, y = pos[node]
                plt.text(x, y - 0.02, node_info, fontsize=font_size, ha='center', va='top',  # Ajustar 'y' para que esté debajo
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.0))  # Hacer la caja transparen
                
        # Add legend
        legend_elements = [
        mpatches.Patch(color='red', label='Mart Domain'),       
        mpatches.Patch(color='yellow', label='Intermediate Domain'),          
        mpatches.Patch(color='#ADD8E6', label='Staging Domain')               
        ]

        plt.legend(handles=legend_elements, 
        loc='upper right', 
        fontsize=15, 
        title="Node Color Legend", 
        title_fontsize=18, 
        frameon=True)

        # Title for the visualization
        plt.title(f"Dependencies for {node_of_interest}", fontsize=50)

        # Display the graph in Streamlit
        st.pyplot(plt)

    # **Functionality 2**
    elif option == "Schedule Optimization":
        st.header("Schedule Optimization")
        # Add multiselect to let user choose the tables to discard
        if df is not None:
            df_staging = df[df['Source_Domain'] == 'staging']
            all_tables = df_staging['Source'].unique().tolist()  
            tables_to_remove = st.multiselect(
                "Select tables to remove",
                options=all_tables,
                default=[]
            )
            if 'df_update_times' not in st.session_state:
                st.session_state['df_update_times'] = pd.DataFrame()

            if 'df_bottlenecks_final' not in st.session_state:
                st.session_state['df_bottlenecks_final'] = df.copy()
            
            selected_table = st.selectbox("Select a table to update its reload time", options=all_tables)
            new_reload_time = st.text_input("Enter new reload time for the selected table (format: HH:MM)", placeholder="e.g., 18:00")

            if st.button("Add Schedule Update"):
                if new_reload_time and selected_table:
                    try:
                        hours, minutes = map(int, new_reload_time.split(':'))
                        if 0 <= hours < 24 and 0 <= minutes < 60:
                            if 'updates' not in st.session_state:
                                st.session_state['updates'] = []
                            st.session_state['updates'].append((selected_table, new_reload_time))
                            st.success(f"Added update: {selected_table} -> {new_reload_time}")
                        else:
                            st.error("Invalid time format. Use HH:MM (24-hour format).")
                    except ValueError:
                        st.error("Invalid time format. Use HH:MM (24-hour format).")

            if 'updates' in st.session_state and st.session_state['updates']:
                st.write("### Pending Updates:")
                updates_df = pd.DataFrame(st.session_state['updates'], columns=["Table", "New Reload Time"])
                st.dataframe(updates_df)

                if st.button("Apply Updates"):
                    st.session_state['df_update_times'] = update_reload_times(df, st.session_state['updates'])
                    st.success("Updates applied successfully!")
                    st.session_state['updates'] = []

        # If you add a CSV, read and show it
        if uploaded_file is not None:    
            # Button to optimize with CSV data
            optimize_from_csv = st.button("Optimize from CSV")

        if optimize_from_csv:
            if df is not None:
                # Store original status from df before deleting tables
                df_bottlenecks_original = df.copy()  
                
                # Optimize original dataframe without modifications
                staging_bottleneck_info, choosed_combined_loads_groups, df_bottlenecks_original_optimized = optimize_csv_data(df_bottlenecks_original)

                # If schedules updates applied update the dataframe to optimize
                if not st.session_state['df_update_times'].empty:
                    st.session_state['df_bottlenecks_final'] = st.session_state['df_update_times'].copy() 

                if tables_to_remove:
                    st.session_state['df_bottlenecks_final'] = remove_tables_by_source(st.session_state['df_bottlenecks_final'], tables_to_remove)
                    st.write("Selected tables have been removed.")

                # Optimize final dataframe (with changes)
                staging_bottleneck_info_final, choosed_combined_loads_groups_final, df_bottlenecks_final_optimized = optimize_csv_data(st.session_state['df_bottlenecks_final'])
                
                st.write("Optimization completed using CSV data.")

                # Staging results expander
                with st.expander("Staging Bottleneck Results"):
                    
                    staging_bottleneck_df = pd.DataFrame.from_dict(staging_bottleneck_info_final, orient='index')
                    staging_bottleneck_df.reset_index(drop=True, inplace=True)
                    staging_bottleneck_df.insert(0, 'Domain', staging_bottleneck_info_final.keys())

                    new_column_names = {
                        'bottleneck_table': 'Bottleneck_Table',
                        'bottleneck_time(hhmm)': 'Bottleneck_Time(HH:MM)'
                    }
                    staging_bottleneck_df.rename(columns=new_column_names, inplace=True)
                    staging_bottleneck_df = staging_bottleneck_df[['Domain','Bottleneck_Table','Bottleneck_Time(HH:MM)']]
                    st.write(staging_bottleneck_df.sort_values(by='Domain'))   

                with st.expander("Load Groups"):
                    
                    combined_loads_with_bottleneck = choosed_combined_loads_groups_final.merge(
                    staging_bottleneck_df[['Domain', 'Bottleneck_Time(HH:MM)']],
                    on='Domain',
                    how='left')

                    cols = combined_loads_with_bottleneck.columns.tolist()
                    cols.insert(1, cols.pop(cols.index('Bottleneck_Time(HH:MM)')))
                    combined_loads_with_bottleneck = combined_loads_with_bottleneck[cols]
                    st.dataframe(combined_loads_with_bottleneck)

                with st.expander("Final Bottleneck Results"):

                    df_sorted = df_bottlenecks_final_optimized.sort_values(by='Domain_Table')
                    df_cleaned = df_sorted.reset_index(drop=True).iloc[:, :-1]

                    new_column_names = {
                        'Domain_Table':'Domain',
                        'table': 'Bottleneck_Table',
                        'total_load_time(hh:mm)': 'Total_Load_Time(HH:MM)'
                    }
                    df_cleaned.rename(columns=new_column_names, inplace=True)
                    df_cleaned = df_cleaned[['Domain','Bottleneck_Table','Total_Load_Time(HH:MM)']]

                    st.dataframe(df_cleaned)

                # Create graphics to compare the original optimizacion versus after changes
                df_bottlenecks_original_optimized["State"] = "Before Updates"
                df_bottlenecks_final_optimized["State"] = "After Updates"

                # Concat Dataframes
                df_combined = pd.concat([df_bottlenecks_original_optimized[["Bottleneck_Time(hours)", "Domain_Table", "State"]],
                                        df_bottlenecks_final_optimized[["Bottleneck_Time(hours)", "Domain_Table", "State"]]])

                # Build comparative bar plot
                fig_combined = px.bar(
                    df_combined,
                    x="Bottleneck_Time(hours)",
                    y="Domain_Table",
                    color="State",  
                    barmode="group",  
                    orientation='h',  
                    labels={"Bottleneck_Time(hours)": "Bottleneck Time (hours)", "Domain_Table": "Domain"},
                    template="plotly_white",
                    color_discrete_map={"Before Updates": "blue", "After Updates": "orange"}
                )

                fig_combined.update_layout(
                    xaxis_title="Bottleneck Time (hours)", 
                    yaxis_title="Domain",
                    legend_title="State"
                )

                # Only show the comparative plot if changes have been applied
                if tables_to_remove or not st.session_state['df_update_times'].empty:
                    with st.expander("Comparison of Bottleneck Time (Before vs After Updates)"):
                        st.plotly_chart(fig_combined, use_container_width=True)

                # Button to show final results if there have been no schedule updates or tables removed
                if not tables_to_remove and st.session_state['df_update_times'].empty:
                    with st.expander("Domain Schedule Results Plot"):
                        df_bottlenecks_final_optimized["Bottleneck_Time(hours)"] = df_bottlenecks_final_optimized["total_load_time(sec)"] / 3600

                        fig = px.bar(
                            df_bottlenecks_final_optimized,
                            x="Bottleneck_Time(hours)",
                            y="Domain_Table",
                            orientation='h',  
                            title="Bottleneck Time by Domain (in Hours)",
                            labels={"Bottleneck_Time(hours)": "Bottleneck Time (hours)", "Domain_Table": "Domain"},
                            template="plotly_white",
                            color_discrete_sequence=["orange"]
                        )
                        fig.update_layout(xaxis_title="Bottleneck Time (hours)", yaxis_title="Domain")
                        
                        st.plotly_chart(fig)

    if option == "Sankey Diagrams":
    # First Diagram (Domain flow dependencies)
        def create_sankey_diagram(df):
            all_nodes = list(pd.concat([df["Source_Domain"], df["Destination_Domain"]]).unique())
            node_indices = {node: i for i, node in enumerate(all_nodes)}

            # Create links (sources -> targets) and calculate their count
            flows = df.groupby(["Source_Domain", "Destination_Domain"]).size().reset_index(name="Count")
            flows["Source_Index"] = flows["Source_Domain"].map(node_indices)
            flows["Target_Index"] = flows["Destination_Domain"].map(node_indices)

            sources = flows["Source_Index"].tolist()
            targets = flows["Target_Index"].tolist()
            values = flows["Count"].tolist()

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            ))

            return fig

        # **Functionality 3**
        st.title("Sankey Diagrams")
        st.write("This first plot is a Sankey interactive diagram that shows domain flow dependencies")

        fig = create_sankey_diagram(df)
        st.plotly_chart(fig, use_container_width=True)

        # Second Diagram (Deployment feeding flow)
        st.write("The second Sankey diagram shows how deployments feed datalake tables")

        df_staging = df[df['Source_Domain'] == 'staging']
        sources = df_staging['Source'].unique()
        sources_with_all = ["All Sources"] + list(sources)
        sources_selected = st.multiselect(
            "Select all, one or more sources",
            options=sources_with_all,
            default=["All Sources"]
        )

        if "All Sources" in sources_selected:
            df_selected_source = df_staging
        else:
            df_selected_source = df_staging[df_staging['Source'].isin(sources_selected)]

        mart_domains = df_selected_source['Mart_Domain'].unique()
        mart_domains_selected = st.multiselect(
            "Select one or more Mart Domains",
            options=mart_domains,
            default=mart_domains[0]
        )

        df_selected_mart_domain = df_selected_source[df_selected_source['Mart_Domain'].isin(mart_domains_selected)]

        # Create paths through columns 'Source', 'Destination', 'Mart_Domain', and 'Mart'
        df_selected_mart_domain['camino'] = df_selected_mart_domain['Source'] + '->' + \
                                            df_selected_mart_domain['Destination'] + '->' + \
                                            df_selected_mart_domain['Mart_Domain'] + '->' + \
                                            df_selected_mart_domain['Mart']

        # Function to calculate bottleneck times using the provided round_time_up function
        bottleneck_times = {}
        for destination in df_selected_mart_domain['Destination'].unique():
            feeding_rows = df_selected_mart_domain[df_selected_mart_domain['Destination'] == destination]
            max_reload_time = feeding_rows['Reload_Time'].max()
            if pd.notna(max_reload_time):  # Ensure no NaN values are processed
                time_hhmm = seconds_to_hhmm(max_reload_time)  # Convert seconds to 'HH:MM'
                rounded_time_hhmm = round_time_up(time_hhmm)  # Round to next quarter-hour
                bottleneck_times[destination] = rounded_time_hhmm

        for mart_domain in df_selected_mart_domain['Mart_Domain'].unique():
            feeding_rows = df_selected_mart_domain[df_selected_mart_domain['Mart_Domain'] == mart_domain]
            max_reload_time = feeding_rows['Reload_Time'].max()
            if pd.notna(max_reload_time):  # Ensure no NaN values are processed
                time_hhmm = seconds_to_hhmm(max_reload_time)  # Convert seconds to 'HH:MM'
                rounded_time_hhmm = round_time_up(time_hhmm)  # Round to next quarter-hour
                bottleneck_times[mart_domain] = rounded_time_hhmm


        # Create nodes and links for Sankey Diagram
        nodos = list(set(df_selected_mart_domain['camino'].str.split('->').explode()))
        nodos_dict = {nodo: i for i, nodo in enumerate(nodos)}

        # Assign colors for each different source (e.g., sap, sp...)
        unique_sources = df_selected_mart_domain['Source'].unique()
        prefijos = {}
        cmap = matplotlib.colormaps.get_cmap("YlGnBu")
        num_colores = len(set(source.split("_")[0] for source in unique_sources if "_" in source))
        colores_paleta = [cmap(i / num_colores) for i in range(num_colores)]

        prefijos_unicos = sorted(set(source.split("_")[0] for source in unique_sources if "_" in source))
        for i, prefijo in enumerate(prefijos_unicos):
            colores_rgb = tuple(int(c * 255) for c in colores_paleta[i][:3])
            prefijos[prefijo] = f"rgb{colores_rgb}"

        colores_nodos = []
        nodos_time_to_load = []
        for nodo in nodos:
            if nodo in df_selected_mart_domain['Source'].unique():  # Nivel 1: Source
                if "_" in nodo:
                    prefijo = nodo.split("_")[0]
                    colores_nodos.append(prefijos.get(prefijo, "skyblue"))
                else:
                    colores_nodos.append("skyblue")

                reload_time = df_selected_mart_domain[df_selected_mart_domain['Source'] == nodo]['Reload_Time'].mean()
                reload_time_hhmm = seconds_to_hhmm(reload_time)
                nodos_time_to_load.append(f"Reload Time: {reload_time_hhmm}")
            elif nodo in df_selected_mart_domain['Destination'].unique():  # Nivel 2: Destination
                colores_nodos.append("white")
                nodos_time_to_load.append(f"Bottleneck Time: {bottleneck_times.get(nodo, 'N/A')}")
            elif nodo in df_selected_mart_domain['Mart_Domain'].unique():  # Nivel 3: Mart_Domain
                colores_nodos.append("orange")
                nodos_time_to_load.append(f"Bottleneck Time: {bottleneck_times.get(nodo, 'N/A')}")
            else:  # Nivel 4: Mart
                colores_nodos.append("orange")
                nodos_time_to_load.append("")

        enlaces = []
        enlaces_median_load_time = []
        for camino in df_selected_mart_domain['camino']:
            partes = camino.split('->')
            for i in range(len(partes) - 1):
                source = nodos_dict[partes[i]]
                target = nodos_dict[partes[i + 1]]
                enlaces.append([source, target])

                if partes[i] in df_selected_mart_domain['Source'].unique():
                    median_load_time = df_selected_mart_domain[df_selected_mart_domain['camino'] == camino]['Median_Load_Time'].mean()
                    hours = int(median_load_time // 3600)
                    minutes = int((median_load_time % 3600) // 60)
                    seconds = int(median_load_time % 60)
                    median_load_time_hhmmss = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    enlaces_median_load_time.append(f"Median Load Time: {median_load_time_hhmmss}")
                else:
                    enlaces_median_load_time.append("")
        
        # Create Sankey Diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodos,
                color=colores_nodos,
                customdata=nodos_time_to_load,
                hovertemplate="Node: %{label}<br>%{customdata}<extra></extra>"
            ),
            link=dict(
                source=[enlace[0] for enlace in enlaces],
                target=[enlace[1] for enlace in enlaces],
                value=[1] * len(enlaces),
                color="#D3D3D3",
                line=dict(color="#D3D3D3", width=0.5),
                customdata=enlaces_median_load_time,
                hovertemplate=(
                    "Flow from %{source.label} to %{target.label}<br>"
                    "%{customdata}<extra></extra>"
                )
            ),
            textfont=dict(
                color="white",  # Cambiar el color del texto
                size=12  # Puedes ajustar el tamaño de la fuente si lo deseas
        )
        ))

        st.plotly_chart(fig)


