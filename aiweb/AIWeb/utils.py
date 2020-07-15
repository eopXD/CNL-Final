import os, sys, json
from typing import Callable, List, Dict, Optional, ClassVar
from collections import OrderedDict, defaultdict
from mako.template import Template
from mako.lookup import TemplateLookup
from django.http import HttpResponse, Http404, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import requires_csrf_token, csrf_exempt
from django.shortcuts import render, redirect
from django.template.context_processors import csrf
from django.conf import settings
from django.views.decorators.cache import never_cache
def MakoExtraOption()->dict:
    return {
        'input_encoding': 'utf-8',
        'output_encoding': 'utf-8',
        'default_filters': ['decode.utf8'],
        'encoding_errors': 'replace',
        'cache_enabled': 'False',
    }
def MakoContextFilter(template_var_in: Optional[Dict]=None):
    template_var_in = template_var_in or {}
    template_var = template_var_in.copy()
    taboo_list = ['self', 'context']
    for taboo in taboo_list:
        if taboo in template_var:
            del template_var[taboo]
    return template_var
def MakoRender(template_string: str, template_var_in: dict, lookup_dir: list=['html']):
    lookup = TemplateLookup(directories=lookup_dir, **MakoExtraOption())
    template_var = MakoContextFilter(template_var_in)
    return Template(
        template_string.strip(), 
        lookup=lookup, **MakoExtraOption()
    ).render_unicode(**template_var)
def MakoRenderFile(template_path: str, template_var_in: dict, lookup_dir: list=['html']):
    lookup = TemplateLookup(directories=lookup_dir, **MakoExtraOption())
    template_var = MakoContextFilter(template_var_in)
    return Template(
        filename=template_path,
        lookup=lookup, **MakoExtraOption()
    ).render_unicode(**template_var)
class MakoManager:
    r'''A configurable mako wrapper'''
    def __init__(self, lookup_main_dir_list):
        if type(lookup_main_dir_list)==type('html'):
            lookup_main_dir_list = [lookup_main_dir_list]
        if type(lookup_main_dir_list)!=type([]):
            lookup_main_dir_list = []
        self.extra_option = MakoExtraOption()
        self.lookup = TemplateLookup(directories=lookup_main_dir_list, **self.extra_option)
    def render(self, tmpl:str, ctx: Optional[Dict]=None):
        r'''render raw string ``tmpl`` in mako'''
        return Template(tmpl,
            lookup=self.lookup, **self.extra_option
            ).render_unicode(**MakoContextFilter(ctx))
    def render_file(self, tmpl_file:str, ctx: Optional[Dict]=None):
        r'''render plain file ``tmpl_file`` in mako
        (notice that there is no lookup)'''
        return Template(filename=tmpl_file,
            lookup=self.lookup, **self.extra_option
            ).render_unicode(**MakoContextFilter(ctx))


