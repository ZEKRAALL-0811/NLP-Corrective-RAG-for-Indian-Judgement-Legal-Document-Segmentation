"""
JurisGraph UI Package
"""
from src.ui.styles import CUSTOM_CSS
from src.ui.components import render_confidence_gauge, get_role_tag_html, get_grade_badge_html
from src.ui.pages.home import render_home_page
from src.ui.pages.upload_query import render_upload_query_page
from src.ui.pages.pipeline import render_pipeline_page
from src.ui.pages.document_analysis import render_doc_analysis_page
from src.ui.pages.dataset_details import render_dataset_page
