# Telemac3D_to_xarray
A script to convert a Selafin file from Telemac 3D into a Xarray file tentatively compliant with CF convention and U-Grid.
## Disclaimer
* This has only be tested on a very **limited** case of Selafin files we use in our studies.
* The reading of Selafin file is possible only if you have properly installed Telemac 3D on your computer. 
We do not provide the scripts as they are not maintained on git but SVN.
## To do
* The remapping variable function should be implemented for **all** the potential variables generated by Telemac into a standard name. 
Support from Telemac to provide the list of Selafin variables [has been asked but not yet answered yet]
(http://www.opentelemac.org/index.php/kunena/scripts/13550-mapping-telemac3d-variables-into-cf-compliant-variables).
* Check that the files are actually close to be compliant with the standards.
