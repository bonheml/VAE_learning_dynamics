from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers


def get_stitching_layer(prev_layer):
    if type(prev_layer) == layers.Dense:
        return layers.Dense(prev_layer.output_shape[1], name="stitch")
    if type(prev_layer) == layers.Conv2D:
        return layers.Conv2D(prev_layer.filters, 1, name="stitch")
    if type(prev_layer) == layers.Conv2DTranspose:
        return layers.Conv2DTranspose(prev_layer.filters, 1, name="stitch")


def stitch_submodel(m1, m2, last_m1, first_m2):
    # Freeze the weights of the pretrained submodels
    m1.trainable = False
    m2.trainable = False

    model = Sequential()
    # Assign the first N layers of m1 to the new submodel
    for layer in m1.layers:
        model.add(layer)
        if layer.name == last_m1:
            # Add the stitching layer
            model.add(get_stitching_layer(layer))
            break

    # Handle the case where stitching occurs right after the sampling layer
    if last_m1 == "sampling":
        return model

    # Assign the last M layers of m2 to the new submodel
    skip = True
    for layer in m2.layers:
        if layer.name == first_m2:
            skip = False
        if skip is False:
            model.add(layer)

    return model


def prepare_pretrained_models(m1_path, m2_path, target, last_m1=None, first_m2=None):
    m1 = load_model(m1_path)
    m2 = load_model(m2_path)
    sub_model_1 = m1.encoder if target == "encoder" else m1.decoder
    sub_model_2 = m2.encoder if target == "encoder" else m2.decoder

    # Handle the case where this is the part we don't stitch
    if last_m1 is None:
        model = sub_model_1 if target == "encoder" else sub_model_2
        model.trainable = False
        return model

    return stitch_submodel(sub_model_1, sub_model_2, last_m1, first_m2)
