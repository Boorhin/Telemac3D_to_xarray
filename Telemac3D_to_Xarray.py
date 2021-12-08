import numpy as np
from datetime import datetime, timedelta
import sys, os, logging
from dask.distributed import Client
logger = logging.getLogger(__name__)
import xarray as xr

try:
    from data_manip.formats.selafin import Selafin
except ImportError:
    # Activating PYTEL
    try:
        pytel = os.path.join(os.environ['HOMETEL'], 'scripts', 'python3')
        if pytel not in sys.path:
            logger.warning('adding telemac python scripts to PYTHONPATH: %s' %
                           pytel)
            sys.path.append(pytel)

        from data_manip.formats.selafin import Selafin
    except (KeyError, ImportError):
        logger.error(
            'Telemac python scripts cannot be found. These are distributed together with the Telemac source code. This reader will not work.'
        )

class T3D_Xr:
    def __init__(self, filename=None, name=None, start_time=None, n_workers=8,
                threads=2, memory_limit='1GB'):
            def vardic(vars_slf):
                """
                Match the selafin variables from Telemac 3D to the variables used in
                OpenDrift.
                """
                # Define all the variables used in OpenDrift as a dictionary
                # This is done to the best of our knowledge
                Vars_OD = {
                    'VELOCITY U      ': 'x_sea_water_velocity',
                    'VELOCITY V      ': 'y_sea_water_velocity',
                    'VELOCITY W      ': 'upward_sea_water_velocity',
                    'TURBULENT ENERGY': 'turbulent_kinetic_energy',
                    'TEMPERATURE     ': 'sea_water_temperature',
                    'SALINITY        ': 'sea_water_salinity',
                    'NUZ FOR VELOCITY': 'ocean_vertical_diffusivity',
                    'NUX FOR VELOCITY': 'horizontal_diffusivity',
                    'ELEVATION Z     ': 'Altitude',
                }

                No_OD_equiv = {
                    'x_wind', 'y_wind', 'wind_speed',
                    'sea_floor_depth_below_sea_level', 'wind_from_direction',
                    'sea_ice_x_velocity', 'sea_ice_y_velocity',
                    'sea_surface_wave_significant_height',
                    'sea_surface_wave_stokes_drift_x_velocity',
                    'sea_surface_wave_stokes_drift_y_velocity',
                    'sea_surface_wave_period_at_variance_spectral_density_maximum',
                    'sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment',
                    'sea_ice_area_fraction', 'surface_downward_x_stress',
                    'surface_downward_y_stress', 'turbulent_generic_length_scale'
                }
                No_Telemac_equiv = {
                    'NUY FOR VELOCITY',
                    'DISSIPATION     ',
                }
                variables = []
                var_idx = []
                for i, var in enumerate(vars_slf):
                    try:
                        variables.append(Vars_OD[var])
                        var_idx.append(i)
                    except:
                        logger.info(
                            "Selafin variable {} has no equivalent in OpenDrift".
                            format(var))
                return np.array(variables), np.array(var_idx)


            def create_array():
                return xr.Dataset(
                attrs={
                    'Conventions': 'CF-1.6',
                    'standard_name_vocabulary': 'CF-1.6',
                    'history': 'Created ' + str(datetime.now()),
                    'source': 'Telemac 3D',
                    },)

            client = Client(n_workers=n_workers, threads_per_worker=threads,
                    memory_limit=memory_limit)

            self.name = name if name is not None else filename
            # self.timer_start("open dataset")
            # logger.info('Opening dataset: %s' % filename)
            self.slf = Selafin(filename)

            self.ds=create_array()
            self.ds.attrs['title']= self.slf.title
            self.ds.attrs['original file']=self.slf.file['name']
            # U-grid conventions
            self.ds["Mesh2D"]=2
            self.ds["Mesh2D"].attrs={
                'cf_role': "mesh_topology",
                'long_name': "Topology data of 2D unstructured mesh" ,
                'topology_dimension': 2 ,
                'node_coordinates': ['node_x', 'node_y'] ,
                'face_node_connectivity': "face_nodes" ,
                'face_dimension': 'face',
                }
            # CF conventions
            self.ds['crs']=0
            self.ds['crs'].attrs={
                'grid_mapping_name':"WGS 84 / UTM zone 30N",
                'semi_major_axis' : 6378137.0, #WGS84
                'inverse_flattening': 298.25722356, #WGS84
                'scale_factor_at_central_meridian': 0.9996,
                'longitude_of_central_meridian': 3,
                'latitude_of_projection_origin': 0,
                'false_easting': 500000,
                'false_northing':0}

            ### time management
            # Slf files have a wrong start time due to a formating error in Telemac
            if start_time is not None:
                if type(start_time) is datetime:
                    self.start_time = start_time
                else:
                    logger.warning(
                        "The specified start time is not a datetime object")
            else:
                logger.info("loading the datetime from the selafin file")
                self.start_time = datetime(self.slf.datetime[0],
                                           self.slf.datetime[1],
                                           self.slf.datetime[2],
                                           self.slf.datetime[3],
                                           self.slf.datetime[4])
            self.times = []
            for i in range(len(self.slf.tags['times'])):
                self.times.append(self.start_time +
                                  timedelta(seconds=self.slf.tags['times'][i]))
            self.ds.coords['time']= ('time', self.times,{
                                    'standard_name': 'time',
                                    'long_name': 'time',
                                    'axis': 'T',
                                    })
            # for OpenDrift
            self.ds.attrs['proj4']='+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs '

            # Building Coordinates
            self.ds.coords['node_x']=(['node'], self.slf.meshx,{
                                'units':'m',
                                'standard_name': 'projection_x_coordinate',
                                'axis': 'X',
                                })
            self.ds.coords['node_y']=(['node'], self.slf.meshy,{
                                'units':'m',
                                'standard_name': 'projection_y_coordinate',
                                'axis': 'Y',
                                })

            self.ds.coords['layer']=('layer', range(self.slf.nplan))

            # We don't know if the indexing is anti-clockwise will trust T3D...
            self.ds['face_nodes']=(['face','max_face_nodes'],
                self.slf.ikle2,{
                    'cf_role': "face_node_connectivity",
                    'long_name': "Maps every triangular face to its three corner nodes",
                    'start_index': 0 })
            self.ds['volume_node_connectivity']=(['cell','max_cell_node'],
                self.slf.ikle3, {
                    'cf_role': "volume_node_connectivity",
                    'long_name': "Maps every volume to its corner nodes.",
                    'start_index': 0})

            self.variables, self.var_idx = vardic(self.slf.varnames)
            # populate the variables
            for i in range(len(self.var_idx)):
                buff= np.empty((len(self.times), self.slf.nplan, self.slf.npoin2),
                    dtype= 'float32')
                for t in range(len(self.times)):
                    buff[t]=self.slf.get_variables_at(t,[i]). \
                        reshape(self.slf.nplan,-1)
                unit=self.slf.varunits[i].strip() # <= match it with CF
                self.ds[self.variables[i]]=(('time','siglay','node'),buff,
                                        {'units':unit,
                                        'grid_mapping' : 'crs'})
            # correct sea-floor sea_floor_depth_below_sea_level

            # self.end_time = self.times[-1]
            # self.altitude_ID=np.array(self.slf.varindex)[np.array( \
            #                     self.slf.varnames)=='ELEVATION Z     '].tolist()
            # self.meshID=(np.arange(self.slf.nplan)[:,None] \
            #              *self.slf.npoin2).astype(int)


            # self.timer_end("build index")
            # self.timer_end("open dataset")

    def write_array(self, filename):
        self.ds.to_zarr(filename)
