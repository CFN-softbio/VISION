# simple_template = """{{- start.plan_name }} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})"""


# count_template = """{% if 'plan_header_override' in start -%}{{start.plan_header_override}}{% else %}{{start.plan_name}}{% endif %} {% if 'sample_savename' in start -%}{{start.sample_savename}}{% else %}{%if 'sample_name' in start -%}{{start.sample_name}}{% endif %}{% endif %} :  {{start.plan_args.num}} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

# Scan Plan
# ----------------------------------------
# {{ start.plan_type }}
# {%- for k, v in start.plan_args | dictsort %}
#     {{ k }}: {{ v }}
# {%-  endfor %}
# {% if 'signature' in start -%}
# Call:
#     {{ start.signature }}
# {% endif %}

# Metadata
# ----------------------------------------
# {% for k, v in start.items() -%}
# {%- if k not in ['plan_type', 'plan_args'] -%}{{ k }} : {{ v }}
# {% endif -%}
# {%- endfor %}


# """


# #single_motor_template = """{{- start.plan_name}} :  {{ start.motors[0]}} {{start.plan_args.start}} {{start.plan_args.stop}} {{start.plan_args.num}} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

# # FIX: remove start and stop motors for now in template. Need to fix
# #single_motor_template = """{% if 'plan_header_override' in start -%}{{start.plan_header_override}}{% else %}{{start.plan_name}}{% endif %} :  {{ start.motors[0]}}  {{'%0.3f' %start.plan_args.start|float}}    {{'%0.3f' %start.plan_args.stop|float}} {{start.plan_args.num}} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

# single_motor_template = """{% if 'plan_header_override' in start -%}{{start.plan_header_override}}{% else %}{{start.plan_name}}{% endif %} :  {{ start.motors[0]}}  {{start.plan_args.num}} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

# Scan Plan
# ----------------------------------------
# {{ start.plan_type }}
# {%- for k, v in start.plan_args | dictsort %}
#     {{ k }}: {{ v }}
# {%-  endfor %}
# {% if 'signature' in start -%}
# Call:
#     {{ start.signature }}
# {% endif %}

# Metadata
# ----------------------------------------
# {% for k, v in start.items() -%}
# {%- if k not in ['plan_type', 'plan_args'] -%}{{ k }} : {{ v }}
# {% endif -%}


# {%- endfor -%}

# """


# from collections import defaultdict
# TEMPLATES = defaultdict(lambda: simple_template)
# TEMPLATES['ct'] = count_template
# TEMPLATES['count'] = count_template
# TEMPLATES['scan'] = single_motor_template
# TEMPLATES['dscan'] = single_motor_template
# TEMPLATES['ascan'] = single_motor_template
# TEMPLATES['ID_calibration'] = single_motor_template

# #from jinja2 import Template # What is this needed for?

# # connect olog
# from functools import partial
# from pyOlog import SimpleOlogClient
# from bluesky.callbacks.olog import logbook_cb_factory

# # Set up the logbook. This configures bluesky's summaries of
# # data acquisition (scan type, ID, etc.).

# LOGBOOKS = ['Data Acquisition']  # list of logbook names to publish to
# simple_olog_client = SimpleOlogClient()
# generic_logbook_func = simple_olog_client.log
# configured_logbook_func = partial(generic_logbook_func, logbooks=LOGBOOKS)

# This is for ophyd.commands.get_logbook, which simply looks for
# a variable called 'logbook' in the global IPython namespace.

##RL comment all below out at 06/11/19 to disable the olog. Expecting the newer version to fix it from DAMA.
# logbook = simple_olog_client


##logbook_cb = logbook_cb_factory(configured_logbook_func)
# logbook_cb = logbook_cb_factory(configured_logbook_func, desc_dispatch=TEMPLATES)

## uncomment this for debugging and comment the next two lines
##RE.subscribe(logbook_cb)
# import nslsii
# nslsii.configure_olog(get_ipython().user_ns, callback=logbook_cb)
