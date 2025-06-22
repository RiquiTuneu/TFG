import pandas as pd

# Function to convert seconds to 'HH:MM' format
def seconds_to_hhmm(seconds):
    """
    Convert seconds to 'HH:MM' format.

    Args:
        seconds (int): Time in seconds.

    Returns:
        str: Time in 'HH:MM' format.
    """
    if isinstance(seconds, float):
        seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"

# Function to round time up to the nearest quarter-hour
def round_time_up(time_str):
    """
    Rounds the given time in 'HH:MM' format up to the nearest quarter-hour after adding 10 minutes.
    
    If the minutes exceed 59 after adding 10, the hours are incremented accordingly, 
    and the minutes are wrapped back to 0-59.

    Args:
        time_str (str): Time string in 'HH:MM' format.

    Returns:
        str: Updated time in 'HH:MM' format rounded to the nearest quarter-hour.
    """
    # Split the time into hours and minutes
    hours, minutes = map(int, time_str.split(":"))

    # Add 10 minutes
    minutes += 10

    # Adjust hours and minutes if minutes overflow past 59
    if minutes >= 60:
        hours += minutes // 60  # Increment hours
        minutes %= 60           # Keep minutes within 0-59

    # Round minutes to the nearest quarter-hour
    if 0 <= minutes <= 14:
        rounded_minutes = 15
    elif 15 <= minutes <= 29:
        rounded_minutes = 30
    elif 30 <= minutes <= 44:
        rounded_minutes = 45
    else:  # 45 <= minutes <= 59
        rounded_minutes = 0
        hours += 1  # Increment the hour if rounding up to the next hour

    # Keep hours within 0-23 for a 24-hour clock
    hours %= 24

    # Format the hours and rounded minutes into 'HH:MM' format
    return f"{hours:02d}:{rounded_minutes:02d}"

# Function to calculate the bottleneck information for each Mart_Domain
def calculate_staging_bottlenecks(df):
    """
    Calculate bottleneck information for each Mart_Domain based on the highest Bottleneck_Time for 'staging' Source_Domain.

    Args:
        df (pd.DataFrame): Input DataFrame containing columns 'Mart_Domain', 'Source_Domain', 'Bottleneck_Time', and 'Source'.
        time_conversion_func (function): Function to convert seconds to 'HH:MM' format.

    Returns:
        dict: Dictionary containing bottleneck information for each Mart_Domain.
    """
    bottleneck_info = {}

    # Iterate over unique Mart_Domains
    for domain in df['Mart_Domain'].unique():
        # Filter for the current domain and 'staging' Source_Domain
        df_domain = df[(df['Mart_Domain'] == domain) & (df['Source_Domain'] == 'staging')]
        # Check if there is valid staging data
        if not df_domain.empty:
            # Identify the row with the maximum Bottleneck_Time
            max_row = df_domain.loc[df_domain['Bottleneck_Time'].idxmax()]

            # Extract relevant information
            bottleneck_info[domain] = {
                'bottleneck_table': max_row['Source'],
                'bottleneck_time(sec)': max_row['Bottleneck_Time'],
                'bottleneck_time(hhmm)': seconds_to_hhmm(int(max_row['Bottleneck_Time']))
            }
        else:
            # Handle cases with no staging data
            bottleneck_info[domain] = {
                'bottleneck_table': None,
                'bottleneck_time(sec)': None,
                'bottleneck_time(hhmm)': None
            }

    return bottleneck_info

# Function to extract intermediate sources
def get_intermediate_sources(df):
    """
    Extracts intermediate sources grouped by Mart_Domain.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Mart_Domain', 'Source_Domain', and 'Source' columns.
    
    Returns:
        dict: Dictionary with Mart_Domain as keys and lists of intermediate sources as values.
    """
    intermediate_sources = {}
    for domain in df['Mart_Domain'].unique():
        df_intermediate = df[
            (df['Mart_Domain'] == domain) & 
            (df['Source_Domain'] == 'intermediate')
        ]
        intermediate_sources[domain] = df_intermediate['Source'].tolist()
    return intermediate_sources

# Function to assign tables to load groups based on bottleneck times
def assign_to_load_groups(df, intermediate_sources, bottleneck_info):
    """
    Assigns intermediate tables to load groups based on bottleneck times.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Mart_Domain' and 'Source' columns.
        intermediate_sources (dict): Dictionary of intermediate sources by Mart_Domain.
        bottleneck_info (dict): Dictionary containing bottleneck information for each Mart_Domain.
    
    Returns:
        tuple: Two dictionaries:
               1. Load groups based on minimum bottleneck time.
               2. Single domain table groups.
    """
    loads_groups = {}
    single_domain_table_groups = {}
    processed_tables = set()

    for domain, tables in intermediate_sources.items():
        for table in tables:
            if table in processed_tables:
                continue
            processed_tables.add(table)

            matching_domains = df[df['Source'] == table]['Mart_Domain'].unique()

            if len(matching_domains) > 1:
                # Tables appearing in more than one domain
                domain_times = [
                    (matching_domain, bottleneck_info[matching_domain]['bottleneck_time(sec)'])
                    for matching_domain in matching_domains
                    if matching_domain in bottleneck_info
                ]

                if domain_times:
                    # Select domain with minimum bottleneck time
                    selected_domain = min(domain_times, key=lambda x: x[1])[0]
                    if selected_domain not in loads_groups:
                        loads_groups[selected_domain] = []
                    loads_groups[selected_domain].append(table)
            else:
                # Tables appearing in only one domain
                single_domain_table = matching_domains[0]
                if single_domain_table not in single_domain_table_groups:
                    single_domain_table_groups[single_domain_table] = []
                single_domain_table_groups[single_domain_table].append(table)

    return loads_groups, single_domain_table_groups

# Function to combine the load groups
def combine_load_groups(loads_groups, single_domain_table_groups):
    """
    Combines the loads_groups and single_domain_table_groups dictionaries into one.
    
    Args:
        loads_groups (dict): Load groups based on minimum bottleneck time.
        single_domain_table_groups (dict): Single domain table groups.
    
    Returns:
        dict: Combined dictionary with all tables grouped under their respective domains.
    """
    combined_loads_groups = loads_groups.copy()  # Make a copy of loads_groups
    for domain, tables in single_domain_table_groups.items():
        if domain not in combined_loads_groups:
            combined_loads_groups[domain] = []  # Create an entry if the domain doesn't exist
        combined_loads_groups[domain].extend(tables)  # Add the tables from single_domain_table_groups
    return combined_loads_groups

# Function to convert HH:MM to seconds
def time_to_seconds(time_str):
    """
    Convert a time string in "HH:MM" format to the equivalent number of seconds.

    Args:
        time_str (str): Time string in "HH:MM" format.

    Returns:
        int: Total number of seconds since midnight, or None if the input is invalid or empty.
    """
    if time_str:  # Check if the input is not empty
        try:
            # Parse the "HH:MM" time string and calculate seconds
            time_obj = pd.to_datetime(time_str, format='%H:%M').time()
            return time_obj.hour * 3600 + time_obj.minute * 60
        except ValueError:
            return None  # Return None if the format is invalid
    return None

# Function to calculate total load time for each table in a given domain
def calculate_table_loaded_times(domain, tables, df, bottleneck_info):
    """
    Calculates the total load time for each table in a given domain.
    
    Args:
        domain (str): The domain to calculate load times for.
        tables (list): List of tables in the domain.
        df (pd.DataFrame): DataFrame containing load time information for tables.
        bottleneck_info (dict): Dictionary with bottleneck information for domains.
    
    Returns:
        list: A list of tuples, each containing a table and its total load time in seconds.
    """
    load_time_list = []
    
    for table in tables:
        # Get table information from the DataFrame
        table_info = df[df['Source'] == table]
        if table_info.empty:
            continue  # Skip if table info is not found
        
        average_load_time = table_info['Median_Load_Time'].values[0]
        
        # Get bottleneck time from the bottleneck info
        if domain in bottleneck_info:
            bottleneck_time = time_to_seconds(bottleneck_info[domain]['bottleneck_time(hhmm)'])
        else:
            continue  # Skip if bottleneck info is not available
        
        # Calculate total load time as the sum of average load time and bottleneck time
        total_load_time = average_load_time + bottleneck_time
        
        # Append the table and its total load time to the list
        load_time_list.append((table, total_load_time))
    
    return load_time_list

# Function to find the bottleneck table and time
def find_max_load_time(load_time_list):
    """
    Finds the table with the maximum total load time from a list of load times.
    
    Args:
        load_time_list (list): List of tuples where each tuple contains a table and its total load time in seconds.
    
    Returns:
        tuple: A tuple containing the table with the maximum load time and the load time in seconds.
    """
    if not load_time_list:
        return None, None  # Return None if the list is empty
    
    # Find the table with the maximum total load time
    max_table, max_total_load_time = max(load_time_list, key=lambda x: x[1])
    return max_table, max_total_load_time

#Function to calculate the intermediate bottleneck info
def calculate_intermediate_bottleneck_info(combined_loads_groups, df, bottleneck_info):
    """
    Calculates bottleneck information for intermediate tables in each domain.
    
    Args:
        combined_loads_groups (dict): Combined dictionary of tables grouped by domains.
        df (pd.DataFrame): DataFrame containing load time information for tables.
        bottleneck_info (dict): Dictionary with bottleneck information for domains.
    
    Returns:
        dict: Dictionary with bottleneck information for each domain.
    """
    intermediates_bottleneck_info = {}

    for domain, tables in combined_loads_groups.items():
        # Calculate load times for all tables in the domain
        load_time_list = calculate_table_loaded_times(domain, tables, df, bottleneck_info)
        
        # Find the table with the maximum load time
        max_table, max_total_load_time = find_max_load_time(load_time_list)
        
        if max_table and max_total_load_time:
            # Convert the maximum load time to 'hh:mm' format
            max_total_load_time_hhmm = seconds_to_hhmm(int(max_total_load_time))
            
            # Store the bottleneck information
            intermediates_bottleneck_info[domain] = {
                'table': max_table,
                'total_load_time(sec)': max_total_load_time,
                'total_load_time(hh:mm)': max_total_load_time_hhmm
            }
    
    return intermediates_bottleneck_info

def remove_tables_by_source(df, tables_to_remove):
    """
    Removes rows from a DataFrame where the values in the 'Source' column match any value in a given list of tables,
    unless the table is the only one within its 'Mart_Domain' with a non-NaN value in the 'Reload_Time' column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        tables_to_remove (list): A list of table names to be removed. Any row in the DataFrame with a 'Source' column 
                                 value matching an entry in this list will be dropped unless it is the only table
                                 in its 'Mart_Domain' with a non-NaN value in the 'Reload_Time' column.

    Returns:
        pd.DataFrame: A new DataFrame with the specified rows removed or retained based on the condition.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Identify rows where 'Source' is in the tables_to_remove list
    rows_to_check = df['Source'].isin(tables_to_remove)

    for index, row in df[rows_to_check].iterrows():
        mart_domain = row['Mart_Domain']

        # Filter for the current Mart_Domain
        domain_rows = df[df['Mart_Domain'] == mart_domain]

        # Check if the current row is the only one with non-NaN 'Reload_Time'
        if domain_rows['Reload_Time'].notna().sum() == 1:
            # Set 'Reload_Time' to 0.0 for this row and skip removing it
            df.at[index, 'Reload_Time'] = 0.0
        else:
            # Remove the row
            df = df.drop(index)

    return df

def optimize_csv_data(df):
    # STEP 1: Calculate staging bottleneck information
    staging_bottleneck_info = calculate_staging_bottlenecks(df)

    # Apply the rounding up function to each entry of hh:mm
    for domain, info in staging_bottleneck_info.items():
        if info['bottleneck_time(hhmm)'] is not None:
            info['bottleneck_time(hhmm)'] = round_time_up(info['bottleneck_time(hhmm)'])

    # STEP 2: Extract intermediate sources
    intermediate_sources = get_intermediate_sources(df)

    # Assign tables to load groups
    loads_groups, single_domain_table_groups = assign_to_load_groups(
        df, 
        intermediate_sources, 
        staging_bottleneck_info
    )

    # Combine the two groups
    combined_loads_groups = combine_load_groups(loads_groups, single_domain_table_groups)

    # Show the chosen single domain table groups
    choosed_combined_loads_groups = pd.DataFrame({
        'Domain': combined_loads_groups.keys(),
        'Tables': [', '.join(tables) for tables in combined_loads_groups.values()] 
    })
    

    # STEP 3: Intermediate bottleneck info
    intermediates_bottleneck_info = calculate_intermediate_bottleneck_info(
        combined_loads_groups, df, staging_bottleneck_info
    )

    # Round total load time and store results
    for domain, info in intermediates_bottleneck_info.items():
        info['total_load_time(hh:mm)'] = round_time_up(info['total_load_time(hh:mm)'])
    
    # Convert to dataframe for final results
    df_bottlenecks_final = pd.DataFrame([
        {'Domain_Table': domain, **info}
        for domain, info in intermediates_bottleneck_info.items()
    ])

    df_bottlenecks_final["Bottleneck_Time(hours)"] = df_bottlenecks_final["total_load_time(sec)"] / 3600
    
    return staging_bottleneck_info, choosed_combined_loads_groups, df_bottlenecks_final

def update_reload_times(df, updates):
    """
    Updates the 'Reload_Time' and 'Bottleneck_Time' for multiple tables in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        updates (list of tuples): A list of tuples, each containing a table name (str) 
                                   and a time string (str) in "HH:MM" format.

    Returns:
        pd.DataFrame: A new DataFrame with the updated 'Reload_Time' and 'Bottleneck_Time'.
    """
    for table_name, time_str in updates:
        # Convert the time to seconds
        new_time_seconds = time_to_seconds(time_str)

        # Update the 'Reload_Time' for the specified table
        df.loc[df['Source'] == table_name, 'Reload_Time'] = new_time_seconds

        # Update the 'Bottleneck_Time' by adding 'Reload_Time' and 'Median_Load_Time'
        df.loc[df['Source'] == table_name, 'Bottleneck_Time'] = (
            df['Reload_Time'] + df['Median_Load_Time']
        )

    return df

# Función para eliminar tablas del DataFrame
def remove_tables_by_source(df, tables_to_remove):
    """
    Elimina filas del DataFrame donde los valores en la columna 'Source' coinciden con algún valor de la lista de tablas
    a menos que la tabla sea la única dentro de su 'Mart_Domain' con un valor no nulo en la columna 'Reload_Time'.
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
        tables_to_remove (list): Lista de nombres de tablas a eliminar. Cualquier fila en el DataFrame cuyo valor en la 
                                 columna 'Source' coincida con un valor de esta lista será eliminada, a menos que sea 
                                 la única tabla en su 'Mart_Domain' con un valor no nulo en la columna 'Reload_Time'.
    
    Returns:
        pd.DataFrame: Un nuevo DataFrame con las filas especificadas eliminadas o retenidas según la condición.
    """
    # Crear una copia del DataFrame para evitar modificar el original
    df = df.copy()

    # Identificar las filas donde 'Source' está en la lista de tablas a eliminar
    rows_to_check = df['Source'].isin(tables_to_remove)

    for index, row in df[rows_to_check].iterrows():
        mart_domain = row['Mart_Domain']

        # Filtrar por el 'Mart_Domain' actual
        domain_rows = df[df['Mart_Domain'] == mart_domain]

        # Comprobar si la fila actual es la única con 'Reload_Time' no nulo
        if domain_rows['Reload_Time'].notna().sum() == 1:
            # Si es la única, establecer 'Reload_Time' a 0.0 y no eliminarla
            df.at[index, 'Reload_Time'] = 0.0

            # Recalculate 'Bottleneck_Time' as the sum of 'Reload_Time' and 'Average_Load_Time'
            median_load_time = df.at[index, 'Median_Load_Time']
            df.at[index, 'Bottleneck_Time'] = 0.0 + median_load_time

        else:
            # Si no es la única, eliminar la fila
            df = df.drop(index)

    return df