import importlib

# General errors and warnings of the Renner program

class RennerError(Exception):
	pass


# Utilities for the Renner program

def loadIntegrals(bas_dict,numprocs):
	try:
		int_mod = importlib.import_module('.{}'.format(bas_dict['type']), package='Integrals')
	except KeyError as err:
		raise RennerError("Basis set type not given! {}".format(err))
	except ImportError as err:
		raise RennerError("Unable to import integrals of the type '{}'! {}".format(bas_dict['type'],err))
	
	return getattr(int_mod,bas_dict['type'])(bas_dict,numprocs)

