# Provided by Biplab Deka @ UIUC

from PIL import Image, ImageDraw
import os
import json

DEVICE_WIDTH = 1440
DEVICE_HEIGHT = 2560
IMAGE_MAX_SIZE = 0.75
PADDING = 3

def has_nonzero_area(bound):
  # print bound
  is_non_zero = (bound[0] < bound[2] and bound[1] < bound[3])
  return is_non_zero

def scale_bounds(node, x_factor, y_factor):
  if node:
    bound = node.get('bounds', [0,0,0,0])
    node['bounds'] = [int(bound[0] * x_factor), int(bound[1] * y_factor),
                         int(bound[2] * x_factor), int(bound[3] * y_factor)]

    for child in node.get('children', []):
      scale_bounds(child, x_factor, y_factor)

def get_elem_bounds(orig_image, element):applied_text_element_bounds
  """Returns bounds for leaf nodes of the view hierarchy."""
  text_bounds = []
  non_text_bounds = []
  if element:
    if element.get('children'):
      for child in element['children']:
        child_bounds = get_elem_bounds(orig_image, child)
        t_bounds, nt_bounds = child_bounds
        text_bounds += t_bounds
        non_text_bounds += nt_bounds
    elif element.get('visible-to-user'):
      text = element.get('text')
      elem_bounds = element.get('bounds')
      # print elem_bounds
      if has_nonzero_area(elem_bounds):
        if text:
          text_bounds.append(elem_bounds)
        else:
          non_text_bounds.append(elem_bounds)
  return (text_bounds, non_text_bounds)

def bounds_to_str(bound):
  """Return bounds as a string for easy comparison."""
  return "{}_{}_{}_{}".format(bound[0], bound[1], bound[2], bound[3])

if __name__ == "__main__":
  import sys
  import time
  ticks = time.time()
  
  input_folder_path = sys.argv[1]
  output_folder_path = sys.argv[2]
  view_ids = [f.split(".")[0] for f in os.listdir(input_folder_path) if ".json" in f]
  view_ids.sort()

  # Comment out this line when you're done testing
  # view_ids = ["0", "1", "2", "3", "4", "5"]

  for view_id in view_ids:
    print "view_id: " + view_id
    try:
      with open(os.path.join(input_folder_path, view_id + ".json")) as data_file:
        view = json.load(data_file)
      image = Image.open(os.path.join(input_folder_path, view_id + ".jpg"))
      width, height = image.size
    except Exception:
      print " --- Skipping Image {} ---".format(view_id)
      continue

    # some images were larger resolution. Rescale those.
    if width > 540:
      image = image.resize((540,960), Image.ANTIALIAS)
    width, height = image.size

    # the view hierarchy has bounds in the pixel dimensions of the screen
    # we have to rescale those to the size of this image
    x_rescale_factor = width/float(DEVICE_WIDTH)
    y_rescale_factor = height/float(DEVICE_HEIGHT)
    scale_bounds(view['activity']['root'], x_rescale_factor, y_rescale_factor)

    text_element_bounds, nontext_element_bounds = get_elem_bounds(image, view['activity']['root'])
    # print "length of text_element_bounds", len(text_element_bounds)
    # print text_element_bounds
    # print "length of nontext_element_bounds", len(nontext_element_bounds)
    # print nontext_element_bounds

    layout_img = Image.new('RGBA', (width, height), 'white')
    draw_layout_img = ImageDraw.Draw(layout_img)

    applied_text_element_bounds = []
    applied_non_text_element_bounds = []

    count_text_element_bounds = 0
    for idx, bound in enumerate(text_element_bounds):
      # We shrink the elements before drawing them to make sure the separations
      # between them are preserved even in the thumbnails that we input to the
      # autoencoder.
      inner_bound = [bound[0] + PADDING, bound[1] + PADDING,
                     bound[2] - PADDING, bound[3] - PADDING]
      if inner_bound[0] >= inner_bound[2] or inner_bound[1] >= inner_bound[3]:
        pass
      #print "blue"
      count_text_element_bounds = count_text_element_bounds + 1
      draw_layout_img.rectangle(inner_bound, fill="blue")
      applied_text_element_bounds.append(bound)


    count_non_text_element_bounds = 0
    for idx, bound in enumerate(nontext_element_bounds):
      # top and bottom strips can be ignored
      if bounds_to_str(bound) == "0_0_540_31" or bounds_to_str(bound) == "0_897_540_960":
        continue

      # Images larger than a certain size are discarded as they are most
      # likely a background image that does not contribute to the UI.
      if (float(bound[2] - bound[0])/width > IMAGE_MAX_SIZE and
          float(bound[3] - bound[1])/height > IMAGE_MAX_SIZE):
        continue

      # We shrink the elements before drawing them to make sure the separations
      # between them are preserved even in the thumbnails that we input to the
      # autoencoder.
      inner_bound = [bound[0] + PADDING, bound[1] + PADDING,
                     bound[2] - PADDING, bound[3] - PADDING]
      if inner_bound[0] >= inner_bound[2] or inner_bound[1] >= inner_bound[3]:
        pass
      #print "red"
      count_non_text_element_bounds = count_non_text_element_bounds + 1
      draw_layout_img.rectangle(inner_bound, fill="red")
      applied_non_text_element_bounds.append(bound)

    # print "length of applied_text_element_bounds", len(applied_text_element_bounds)
    # print applied_text_element_bounds
    # print "length of applied_nontext_element_bounds", len(applied_non_text_element_bounds)
    # print applied_non_text_element_bounds

    layout_img.save(os.path.join(output_folder_path, view_id + "_layout.jpg"))
    with open(os.path.join(output_folder_path, view_id + "_element_count.txt"), 'w') as f:
      f.write("text:")
      f.write(str(count_text_element_bounds))
      f.write("\n")
      f.write("non_text:")
      f.write(str(count_non_text_element_bounds))
      f.write("\n")
  
  print "time diff in seconds", (time.time() - ticks)