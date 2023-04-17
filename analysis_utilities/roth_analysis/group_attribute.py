import numpy as np

def group_attribute(object_list, *args, dict_key = None, secondary_key = None):
    attributes = list(args)
    field_dict = dict()
    
    #Figure out the shape of each attribute
    first_object = object_list[0]
    
    for attr in attributes:
        #if attribute is a dictionary and no specific keys have been given
        if (dict_key == None or (type(dict_key) == list)) and type(getattr(first_object, attr)) == dict :
            raise ValueError("Must provide a single dictionary key.")
        
        #if attribute is a dictionary and a specific key has been given
        if dict_key != None and type(getattr(first_object, attr)) == dict:
            #Shape is (Num Objects, Num Datapoints)
            temp_attr = getattr(first_object, attr)[dict_key]
            
            #if a secondary key is given
            if type(temp_attr) == dict and secondary_key != None:
                
                field_dict[attr] = np.empty((len(object_list), len(temp_attr[secondary_key]))) * np.nan

                for i, obj in enumerate(object_list):
                    field_dict[attr][i] = getattr(obj, attr)[dict_key]
                    
            #No secondary key given but one is required
            elif type(temp_attr) == dict and secondary_key == None:
                raise ValueError(f"Attribute {dict_key} is a dictionary and requires a secondary_key.")
                
            #No secondary key required
            else:
                if type(temp_attr) not in [list, type(np.array([0]))]:
                    temp_attr = [temp_attr]
                    
                field_dict[attr] = np.empty((len(object_list), len(temp_attr))) * np.nan

                for i, obj in enumerate(object_list):
                    field_dict[attr][i] = getattr(obj, attr)[dict_key]
                
        #if attribute is not a dictionary
        if type(getattr(first_object, attr)) != dict:
            #Shape is (Num Objects, Num Datapoints)
            field_dict[attr] = np.empty((len(object_list), len(getattr(first_object, attr)))) * np.nan
            
            for i, obj in enumerate(object_list):
                field_dict[attr][i] = getattr(obj, attr)
        
    return field_dict
