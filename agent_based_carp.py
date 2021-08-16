# importing packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
import random
import itertools
from tqdm import tqdm
import json
from joblib import Parallel, delayed

class AsianCarp:
    def __init__(self, density, distance=100, shoreline_multiplier=0.5, dike_multiplier=0.0625, movement_type=None, upstream_flag=True):
        """
        Parent class of the Asian Carp. Should be designed to be further subclassed
        into species for specific characteristics. Should be designed to work for 
        any water body as a map
        """
        # for shoreline calculations
        self.shoreline_multiplier = shoreline_multiplier

        # for dike calculations
        self.dike_multiplier = dike_multiplier
        
        # carp characteristics
        self.density = density
        self.distance = distance

        # define the movement characteristics, normalized if values are provided
        if not movement_type:
            self.movement_type = {"random":1.0, "autocorrelation":0, "upstream":0}
        else:
            # normalizing
            total_sum = sum(movement_type.values())
            self.movement_type = {key:movement_type[key]/total_sum for key in movement_type}
        self.upstream_flag = upstream_flag

        # The actual place where the agents are stored, which is set within the water map
        self.num_agents = 0
        self.agents = {}
        self.prev_dir = {}

    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value
    
    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, agents_dict):
        if len(agents_dict) != self.num_agents:
            raise ValueError("Length of agent dict not equal to num_agents")
        self._agents = agents_dict

    @property
    def prev_dir(self):
        return self._prev_dir

    @prev_dir.setter
    def prev_dir(self, prev_dict):
        if len(prev_dict) != self.num_agents:
            raise ValueError("Length of prev dict not equal to num_agents")
        self._prev_dir = prev_dict

            
class MSRiver(gpd.GeoDataFrame):
    def __init__(self, asianCarp, *args, spawn_area=None, upstream = None, depth=18.3, shoreline_distance= 100, dike_distance=50, bc_respawn=True, dikes=None, **kwargs):
        """
        The body of water in which the carp will do the simulation. All of the movement 
        rules are defined within the body of water so that the carp can be placed into 
        different environments with different movement rules. This is a subclass of 
        the geopandas dataframe.
        """
        # initializing the geopandas superclass
        super().__init__(*args, **kwargs)

        # recording water body features
        self._shoreline = self['geometry'][0].exterior # shoreline of the actual river
        self._shoreline_buffer = self.shoreline.buffer(shoreline_distance) # buffer of the shoreline based on the distance
        self._dikes = dikes # point locations of the dikes
        if self.dikes is None: # if there are no dikes, make it a blank polygon
            self._dikes_buffer = Polygon()
        else: # otherwise, create a box around the point
            def box_around_point(p, side_length):
                p_coords = np.array(p.coords).flatten()
                minx, miny, maxx, maxy = p_coords[0] - side_length/2, p_coords[1] - side_length/2, p_coords[0] + side_length/2, p_coords[1] + side_length/2
                return box(minx, miny, maxx, maxy)
            self.dikes["geometry"] = [box_around_point(dikes["geometry"][i], dike_distance) for i in range(len(dikes))]
            self._dikes_buffer = self.dikes["geometry"].unary_union
        self._upstream = upstream # record the upstream direction (as a shapely linestring)

        # recording BC spillway features
        self._bc_border = self['geometry'][1].intersection(self.shoreline)
        self.bc_respawn = bc_respawn 

        # polygon of the MS river
        poly_gdf = self['geometry'][0]

        # recording start location, different spawning locations
        self._spawn_area = spawn_area
        if not poly_gdf.intersects(self.spawn_area['geometry'][2]):
            raise ValueError("spawn location not within water body")

        self._agent_bounds = poly_gdf.intersection(self.spawn_area['geometry'][2])

        self.spawn_area['geometry'][0] = self.spawn_area['geometry'][0].intersection(self.agent_bounds)
        self.spawn_area['geometry'][1] = self.spawn_area['geometry'][1].intersection(self.agent_bounds)

        # recording the carp, important for extracting the movement parameters
        self.asianCarp = asianCarp
        
        # initializing the number of carp
        area = self.agent_bounds.area
        volume = area*depth
        self.asianCarp.num_agents = int(volume*self.asianCarp.density)
        print(self.asianCarp.num_agents)

        # initializing the previous direction dictionary
        self.asianCarp.prev_dir = {i:None for i in range(self.asianCarp.num_agents)}

        # initializing the current location dictionary
        self.asianCarp.agents = {i:[] for i in range(self.asianCarp.num_agents)}
        self.respawnAgents()

    @property
    def shoreline(self):
        return self._shoreline

    @property
    def shoreline_buffer(self):
        return self._shoreline_buffer

    @property
    def dikes(self):
        return self._dikes

    @property
    def dikes_buffer(self):
        return self._dikes_buffer

    @property
    def bc_border(self):
        return self._bc_border

    @property
    def upstream(self):
        return self._upstream

    @property
    def spawn_area(self):
        return self._spawn_area

    @property
    def agent_bounds(self):
        return self._agent_bounds

    def update(self, loc_array, prev_array):
        """
        Updates the agents' position for a specific timestep inplace. Also returns the 
        dictionary containing the agents
        """
        # defining the movement behaviors (more streamlined without the multipliers)
        def randomMovement(rand):
            return np.array([self.asianCarp.distance*np.cos(2*np.pi*rand), self.asianCarp.distance*np.sin(2*np.pi*rand)])

        def prevMovement(prev_dir):
            return self.asianCarp.distance*prev_dir

        def upstreamMovement(curr_loc, upstream, eps = 1.0):
            proj_dist = upstream.project(Point(curr_loc.tolist()))
            point_1 = upstream.interpolate(proj_dist)
            point_2 = upstream.interpolate(proj_dist + eps)

            # getting the direction vector
            dir_vector = np.array(point_2).flatten() - np.array(point_1).flatten()
            dir_vector = dir_vector/np.linalg.norm(dir_vector)

            return self.asianCarp.distance*dir_vector


        # getting the boundaries
        poly_gdf = self['geometry'].unary_union
        exterior_gdf = poly_gdf.exterior
        exterior_buffer = exterior_gdf.buffer(10.0)

        curr_loc = loc_array

        if curr_loc[0] is None:
            return curr_loc, prev_array
        
        ### IDEAL TRAVEL ###
        # getting random and upstream movements
        rand = np.random.rand()
        random_movement = randomMovement(rand)
        upstream = upstreamMovement(curr_loc, self.upstream)

        # getting previous direction
        if prev_array is None:
            prev_dir = self.asianCarp.movement_type["random"]*random_movement + self.asianCarp.movement_type["upstream"]*upstream*(2*self.asianCarp.upstream_flag-1)
            prev_dir = prev_dir/np.linalg.norm(prev_dir)
        else:
            prev_dir = prev_array
        
        # getting movement vector for previous direction
        prev_movement = prevMovement(prev_dir)
        
        # saving the ideal travel vector for next timestep
        init_dir = self.asianCarp.movement_type["random"]*random_movement + self.asianCarp.movement_type["autocorrelation"]*prev_movement + self.asianCarp.movement_type["upstream"]*upstream*(2*self.asianCarp.upstream_flag-1)
        # changing the shoreline and dike multipliers to here
        if not self.dikes is None:
            if self.shoreline_buffer.contains(Point(curr_loc.tolist())) and self.dikes_buffer.contains(Point(curr_loc.tolist())):
                init_dir *= self.asianCarp.dike_multiplier
            elif self.shoreline_buffer.contains(Point(curr_loc.tolist())):
                init_dir *= self.asianCarp.shoreline_multiplier
        else:
            if self.shoreline_buffer.contains(Point(curr_loc.tolist())):
                init_dir *= self.asianCarp.shoreline_multiplier

        init_loc = init_dir + curr_loc
        prev_dir_final = init_dir/np.linalg.norm(init_dir)

        transport_line = LineString([curr_loc.tolist(), init_loc.tolist()])
        inter_points = transport_line.intersection(exterior_gdf)

        if poly_gdf.contains(Point(init_loc.tolist())) and inter_points.is_empty:
            new_loc = init_loc
        
        ### NON-IDEAL TRAVEL W/ BINARY SEARCH ###
        else:
            search_flag = True
            lower = 0
            upper = 1
            while search_flag:
                
                frac = (lower + upper)/2

                new_dir = frac*init_dir
                loc = new_dir + curr_loc

                transport_line = LineString([curr_loc.tolist(), loc.tolist()])
                inter_points = transport_line.intersection(exterior_gdf)

                if exterior_buffer.contains(Point(loc.tolist())) and poly_gdf.contains(Point(loc.tolist())) and inter_points.is_empty:
                    new_loc = loc
                    search_flag = False

                elif not exterior_buffer.contains(Point(loc.tolist())) and poly_gdf.contains(Point(loc.tolist())) and inter_points.is_empty:
                    lower = frac

                elif not poly_gdf.contains(Point(loc.tolist())) or not inter_points.is_empty:
                    upper = frac
                if frac <= 0.01:
                    new_loc = loc_array
                    search_flag = False

        return new_loc, prev_dir_final

    def updateAgents(self):
        """
        Function that takes the current agent position dictionary and updates all of the 
        positions based on the movement characteristics.

        Updated in 0804 to include parallelization support via joblib

        Returns None
        """
        
        curr_dict = self.asianCarp.agents.copy()
        prev_dict = self.asianCarp.prev_dir.copy()

        update_list = Parallel(n_jobs=4, prefer="threads")(delayed(self.update)(curr_dict[i], prev_dict[i]) for i in range(len(curr_dict)))

        for i in range(len(curr_dict)):
            curr_dict[i], prev_dict[i] = update_list[i]
        
        self.asianCarp.agents = curr_dict
        self.asianCarp.prev_dir = prev_dict


    def respawn(self, loc_array):
        """
        Function that takes in a carp position (as an array) and decides whether
        it needs to be respawned based on the water body characteristics

        returns (new location, T/F if in the BC spillway, T/F if outside the allowable area)
        """
        def get_random_point_in_polygon(poly):
            """
            Function that takes in a polygon and samples a point inside the polygon as 
            an array. 
            """
            minx, miny, maxx, maxy = poly.bounds
            while True:
                p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(p):
                    return np.array(p.coords).flatten()
        # if it is the initial spawn
        if len(loc_array) == 0:
            return (get_random_point_in_polygon(self.agent_bounds), False, False)
        # if the carp has been taken out of the simulation
        if loc_array[0] is None:
            return ([None, None], False, False)
        loc = Point(loc_array)
        # respawn if it's in the BC spillway
        if self['geometry'][1].contains(loc) and not self['geometry'][0].contains(loc):
            if self.bc_respawn:
                return (get_random_point_in_polygon(self.spawn_area['geometry'][0].intersection(self.spawn_area['geometry'][2])), True, True)
            else:
                return ([None, None], True, True)
        # respawn if the carp is outside the allowable area
        elif not self.spawn_area['geometry'][2].contains(loc):
            distances_to_spawn_areas = self.spawn_area.distance(loc)
            poly_index = np.argmax(distances_to_spawn_areas)
            return (get_random_point_in_polygon(self.spawn_area['geometry'][poly_index].intersection(self.spawn_area['geometry'][2])), False, True)
        # do not respawn
        else:
            return (loc_array, False, False)

    def respawnAgents(self):
        """
        Controls whether a specific agent should be respawned according to the rules 
        set by the water body. Super important that these rules be defined within the 
        water body.

        Updated in 0804 to support parallelization
        """
        update_list = Parallel(n_jobs=4, prefer="threads")(delayed(self.respawn)(self.asianCarp.agents[agent]) for agent in range(len(self.asianCarp.agents)))

        count = 0
        for agent in range(len(self.asianCarp.agents)):
            self.asianCarp.agents[agent], crossing_flag, respawn_flag = update_list[agent]
            count += crossing_flag
            if respawn_flag:
                self.asianCarp.prev_dir[agent] = None

        return self.asianCarp.agents, count

    def plotRiver(self, ax, title, figname=""):
        """
        Method for plotting the river geometry, the dikes, and the individual carp
        """
        ax.cla()
        ax.set_xlim(742000, 752500)
        ax.set_ylim(3.3175e6, 3.3265e6)
        X = [self.asianCarp.agents[i][0] for i in self.asianCarp.agents]
        Y = [self.asianCarp.agents[i][1] for i in self.asianCarp.agents]

        ax.plot(X, Y, 'ro', ms=1)
        self.plot(ax=ax)
        if not self.dikes is None:
            self.dikes.plot(ax=ax, color="green")

        ax.set_title(title)

        if figname:
            fig.savefig(figname)


def create_grid(gdf, side_length=100):
    """
    Takes in a geodataframe, creates a grid (geodataframe) of cells that intersect
    the geodataframe

    Input:
    - gdf, geodataframe used to create the grid
    - side_length, length of a side of each of the cells in meters
    """
    total_bounds = gdf['geometry'].total_bounds
    
    geom = [Polygon([(i, j), (i+side_length, j), (i+side_length, j+side_length), (i, j+side_length)]) for i, j in itertools.product(np.arange(total_bounds[0], total_bounds[2], side_length), np.arange(total_bounds[1], total_bounds[3], side_length))]
    d = {'geometry':geom}
    grid_gdf = gpd.GeoDataFrame(d, crs=gdf.crs)
    
    poly_gdf = gdf['geometry'].unary_union
    
    inter = grid_gdf['geometry'].intersects(poly_gdf)
    
    return grid_gdf[inter]

if __name__ == "__main__":
    # setting up MS River + BC spillway
    # this is a geodataframe that only contains the geometries
    lakes = gpd.read_file("shapefiles/lakes/lakes.shp")
    lakes = lakes.cx[:, :0]
    bonnet_carre = gpd.read_file("shapefiles/bonnet_carre.geojson")
    bonnet_carre['ID'] = 1
    bonnet_carre = bonnet_carre[['ID', 'geometry']]
    full_water = lakes.append(bonnet_carre, ignore_index=True)
    full_water.crs = "EPSG:4326"
    full_water = full_water.to_crs("EPSG:32615")

    # setting up upstream
    # shapely geometry (NOT geodataframe)
    upstream = gpd.read_file("shapefiles/upstream_river.geojson")
    upstream.crs = "EPSG:4326"
    upstream = upstream.to_crs("EPSG:32615")
    upstream = upstream.iat[0, 0]

    ### INPUT NAME OF SPECIES ###
    carp_species = "silver"

    # loading parameters
    with open("carp.json", "r") as openfile:
        carp_dict = json.load(openfile)
    
    # density parameters
    density = carp_dict[carp_species]["density"]
    school_size = 10
    if school_size:
        density /= school_size

    # number of timesteps
    timestep_length = 1 # in hours
    simulation_length = 2*24 #in hours
    timesteps = int(simulation_length/timestep_length)

    # movement multipliers
    shoreline_multiplier = 0.25
    dike_multiplier = 0.0625

    # spawn area
    spawn_area = gpd.read_file('shapefiles/spawn_area_2km.geojson', crs="EPSG:4326").to_crs("EPSG:32615")

    # loading dikes
    dikes = gpd.read_file("shapefiles/dikes_final.geojson")
    dikes.crs = "EPSG:4326"
    dikes = dikes.to_crs("EPSG:32615")
    dike_distance = 100

    # plotting
    PLOT = False

    columns = ["species", "number of carp", "season", "dike config"]
    columns.extend([f'timestep {t+1}' for t in range(timesteps)])
    df_results = pd.DataFrame(columns=columns)

    # iterating over the seasons
    for season in range(4):
        carp_speed = carp_dict[carp_species]["speed"][season]
        distance = carp_speed*(timestep_length*60*60)
        movement_type = carp_dict["movement"][season]
        upstream_flag = carp_dict["upstream"][season]

        print(distance)

        # iterating over the dike configuration
        dike_config = [None, 0, 1, 2, 3]
        for d in dike_config:
            if d is None:
                dikes_final = None
            else:
                dikes_final = dikes.iloc[[d]].reset_index(drop=True)

            # initializing the carp
            asianCarp = AsianCarp(density, distance=distance, shoreline_multiplier=shoreline_multiplier, dike_multiplier=dike_multiplier, movement_type=movement_type, upstream_flag=upstream_flag)

            # initializing MS River environment
            print("Loading simulation environment...")
            ms_dict = {'geometry':full_water['geometry'].tolist()}
            ms_river = MSRiver(asianCarp, ms_dict, spawn_area=spawn_area, upstream=upstream, crs="EPSG:32615", bc_respawn=False, dikes=dikes_final, dike_distance=dike_distance)
            print("Starting simulation")

            # list the contains the number of crossings per timestep
            crossings = []

            if PLOT:
                fig, ax = plt.subplots()
                ms_river.plotRiver(fig, ax, "Start")
                # Note that using time.sleep does *not* work here!
                plt.pause(5)

            df_row = [carp_species, asianCarp.num_agents, season, d]
            for t in tqdm (range (timesteps), 
                            desc="Running…", 
                            ascii=False, ncols=75):
                ms_river.updateAgents()
                _, crossing_count = ms_river.respawnAgents()

                crossings.append(crossing_count)
                
                if PLOT:
                    ms_river.plotRiver(fig, ax, "timestep {}".format(t+1))
                    plt.pause(0.1)
            df_row.extend(crossings)

            df_results.loc[len(df_results.index)] = df_row

            # for testing
            if len(df_results) == 2:
                print(df_results)

    df_results.to_excel(f"output/results_{carp_species}.xlsx")

        
