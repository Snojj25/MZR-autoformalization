"""
HTML Report Generator for Autoformalization Pipeline Results
"""
import os
import json
from datetime import datetime
from typing import Dict, List
from config import config


def generate_html_report(result: Dict, output_path: str = None) -> str:
    """
    Generate an HTML report from pipeline results
    
    Args:
        result: Single result dictionary from pipeline.process()
        output_path: Optional path to save the report
        
    Returns:
        Path to the generated HTML file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.RESULTS_DIR, f"report_{timestamp}.html")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    html_content = _build_html(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def _build_html(result: Dict) -> str:
    """Build the complete HTML document"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autoformalization Pipeline Report</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding: 10px;
            background: #ecf0f1;
            border-left: 4px solid #3498db;
        }}
        
        .header-info {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        
        .badge-success {{
            background: #2ecc71;
            color: white;
        }}
        
        .badge-failure {{
            background: #e74c3c;
            color: white;
        }}
        
        .badge-info {{
            background: #3498db;
            color: white;
        }}
        
        .problem-statement {{
            background: #fff9e6;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        
        .rag-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .rag-table th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        .rag-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        .rag-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .collapsible {{
            margin: 15px 0;
        }}
        
        .collapsible-header {{
            background: #3498db;
            color: white;
            padding: 12px 15px;
            cursor: pointer;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        
        .collapsible-header:hover {{
            background: #2980b9;
        }}
        
        .collapsible-header.success {{
            background: #2ecc71;
        }}
        
        .collapsible-header.success:hover {{
            background: #27ae60;
        }}
        
        .collapsible-header.failure {{
            background: #e74c3c;
        }}
        
        .collapsible-header.failure:hover {{
            background: #c0392b;
        }}
        
        .collapsible-content {{
            display: none;
            padding: 20px;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }}
        
        .collapsible-content.active {{
            display: block;
        }}
        
        .code-block {{
            margin: 15px 0;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .error-box {{
            background: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        
        .error-box pre {{
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-card h3 {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 20px;
        }}
        
        .tactic-badge {{
            display: inline-block;
            background: #9b59b6;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {_format_header(result)}
        {_format_rag_examples(result.get('rag_examples', []))}
        {_format_iterations(result.get('iterations', []))}
        {_format_proof_attempts(result)}
        {_format_metrics(result)}
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <script>
        document.querySelectorAll('.collapsible-header').forEach(header => {{
            header.addEventListener('click', function() {{
                const content = this.nextElementSibling;
                const isActive = content.classList.contains('active');
                
                // Close all in same parent
                const parent = this.closest('.collapsible').parentElement;
                parent.querySelectorAll('.collapsible-content').forEach(c => {{
                    c.classList.remove('active');
                }});
                
                // Toggle current
                if (!isActive) {{
                    content.classList.add('active');
                }}
            }});
        }});
    </script>
</body>
</html>
"""


def _format_header(result: Dict) -> str:
    """Format the header section with problem statement and status badges"""
    nl = result.get('natural_language', 'N/A')
    comp_success = result.get('compilation_success', False)
    proof_success = result.get('proof_success', False)
    total_time = result.get('total_time', 0)
    iterations = result.get('total_iterations', 0)
    
    comp_badge = 'badge-success' if comp_success else 'badge-failure'
    comp_text = '✓ Success' if comp_success else '✗ Failed'
    
    proof_badge = 'badge-success' if proof_success else 'badge-failure'
    proof_text = '✓ Success' if proof_success else '✗ Failed'
    
    return f"""
    <h1>Autoformalization Pipeline Report</h1>
    <div class="header-info">
        <span class="badge {comp_badge}">Compilation: {comp_text}</span>
        <span class="badge {proof_badge}">Proof: {proof_text}</span>
        <span class="badge badge-info">Iterations: {iterations}</span>
        <span class="badge badge-info">Time: {total_time:.2f}s</span>
    </div>
    <div class="problem-statement">
        <strong>Problem Statement:</strong><br>
        {_escape_html(nl)}
    </div>
    """


def _format_rag_examples(examples: List[Dict]) -> str:
    """Format RAG examples as a table"""
    if not examples:
        return ""
    
    rows = ""
    for idx, example in enumerate(examples, 1):
        nl = example.get('natural_language', 'N/A')
        similarity = example.get('similarity', 0)
        rows += f"""
        <tr>
            <td>{idx}</td>
            <td>{_escape_html(nl)}</td>
            <td>{similarity:.3f}</td>
        </tr>
        """
    
    return f"""
    <h2>Step 1: RAG Retrieval</h2>
    <table class="rag-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Natural Language</th>
                <th>Similarity Score</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def _format_iterations(iterations: List[Dict]) -> str:
    """Format compilation iterations as collapsible sections"""
    if not iterations:
        return ""
    
    sections = ""
    for iter_data in iterations:
        iter_num = iter_data.get('iteration', 0)
        code = iter_data.get('formal_statement', '')
        success = iter_data.get('compilation_success', False)
        errors = iter_data.get('errors', [])
        error_cats = iter_data.get('error_categories', {})
        
        header_class = 'success' if success else 'failure'
        status_icon = '✓' if success else '✗'
        status_text = 'Success' if success else 'Failed'
        
        error_section = ""
        if errors:
            error_list = "\n".join([f"<li>{_escape_html(e)}</li>" for e in errors[:5]])
            error_section = f"""
            <div class="error-box">
                <strong>Errors:</strong>
                <ul style="margin-top: 10px; padding-left: 20px;">
                    {error_list}
                </ul>
            </div>
            """
        
        if error_cats:
            cat_text = ", ".join(error_cats.keys())
            error_section += f'<div style="margin-top: 10px; color: #7f8c8d;"><strong>Error Types:</strong> {cat_text}</div>'
        
        sections += f"""
        <div class="collapsible">
            <div class="collapsible-header {header_class}">
                <span><strong>Iteration {iter_num}</strong> - {status_icon} {status_text}</span>
                <span>▼</span>
            </div>
            <div class="collapsible-content">
                <div class="code-block">
                    <pre><code class="language-lean">{_escape_html(code)}</code></pre>
                </div>
                {error_section}
            </div>
        </div>
        """
    
    return f"""
    <h2>Step 2-3: Iterative Formalization</h2>
    {sections}
    """


def _format_proof_attempts(result: Dict) -> str:
    """Format proof attempts as collapsible sections"""
    if not result.get('compilation_success'):
        return '<h2>Step 4: Proof Generation</h2><p><em>Skipped (compilation failed)</em></p>'
    
    proof_attempts = result.get('proof_attempts', [])
    if not proof_attempts:
        return '<h2>Step 4: Proof Generation</h2><p><em>No proof attempts recorded</em></p>'
    
    final_statement = result.get('final_statement', '')
    proof_tactic = result.get('proof_tactic', None)
    proof_success = result.get('proof_success', False)
    
    sections = ""
    for idx, attempt in enumerate(proof_attempts, 1):
        tactic = attempt.get('tactic', 'unknown')
        success = attempt.get('success', False)
        errors = attempt.get('errors', [])
        proof_code = attempt.get('proof_code', '')
        
        # If no proof_code in attempt, try to construct it
        if not proof_code:
            # Try to construct from final_statement if this is the last successful attempt
            if final_statement and success and idx == len(proof_attempts):
                proof_code = final_statement
            # Otherwise, try to construct from the original statement and tactic
            elif final_statement and not success:
                # Extract theorem declaration (remove ':= by sorry' or any proof)
                theorem_decl = final_statement.split(':= by')[0].strip()
                if theorem_decl:
                    proof_code = f"{theorem_decl} := by {tactic}"
        
        header_class = 'success' if success else 'failure'
        status_icon = '✓' if success else '✗'
        status_text = 'Success' if success else 'Failed'
        
        error_section = ""
        if errors:
            error_list = "\n".join([f"<li>{_escape_html(e)}</li>" for e in errors[:5]])
            error_section = f"""
            <div class="error-box">
                <strong>Errors:</strong>
                <ul style="margin-top: 10px; padding-left: 20px;">
                    {error_list}
                </ul>
            </div>
            """
        
        # Highlight the successful tactic
        tactic_display = tactic
        if success and proof_tactic == tactic:
            tactic_display = f'{tactic} <span class="tactic-badge">SUCCESS</span>'
        
        sections += f"""
        <div class="collapsible">
            <div class="collapsible-header {header_class}">
                <span><strong>Attempt {idx}: Tactic <code>{tactic_display}</code></strong> - {status_icon} {status_text}</span>
                <span>▼</span>
            </div>
            <div class="collapsible-content">
                {f'<div class="code-block"><pre><code class="language-lean">{_escape_html(proof_code)}</code></pre></div>' if proof_code else '<p><em>No code available</em></p>'}
                {error_section}
            </div>
        </div>
        """
    
    return f"""
    <h2>Step 4: Proof Generation</h2>
    {sections}
    """


def _format_metrics(result: Dict) -> str:
    """Format metrics as cards"""
    total_time = result.get('total_time', 0)
    iterations = result.get('total_iterations', 0)
    comp_success = result.get('compilation_success', False)
    proof_success = result.get('proof_success', False)
    
    return f"""
    <h2>Metrics Summary</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Total Time</h3>
            <div class="value">{total_time:.2f}s</div>
        </div>
        <div class="metric-card">
            <h3>Iterations</h3>
            <div class="value">{iterations}</div>
        </div>
        <div class="metric-card">
            <h3>Compilation</h3>
            <div class="value">{'✓' if comp_success else '✗'}</div>
        </div>
        <div class="metric-card">
            <h3>Proof</h3>
            <div class="value">{'✓' if proof_success else '✗'}</div>
        </div>
    </div>
    """


def _escape_html(text: str) -> str:
    """Escape HTML special characters"""
    if not text:
        return ""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def generate_batch_html_report(results: List[Dict], output_path: str = None) -> str:
    """
    Generate an HTML report for batch results
    
    Args:
        results: List of result dictionaries
        output_path: Optional path to save the report
        
    Returns:
        Path to the generated HTML file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.RESULTS_DIR, f"batch_report_{timestamp}.html")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # For batch reports, create a summary page with links to individual reports
    html_content = _build_batch_html(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def _build_batch_html(results: List[Dict]) -> str:
    """Build HTML for batch results"""
    from src.utils import calculate_metrics
    
    metrics = calculate_metrics(results)
    
    # Generate individual reports and collect paths
    individual_reports = []
    for idx, result in enumerate(results):
        report_path = generate_html_report(result, 
            os.path.join(config.RESULTS_DIR, f"report_{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"))
        individual_reports.append((idx+1, report_path, result.get('natural_language', 'N/A')[:60]))
    
    report_links = "\n".join([
        f'<li><a href="{path}" target="_blank">Problem {num}: {_escape_html(nl)}...</a></li>'
        for num, path, nl in individual_reports
    ])
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Autoformalization Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        ul {{
            list-style: none;
            padding: 0;
        }}
        ul li {{
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        ul li a {{
            color: #3498db;
            text-decoration: none;
        }}
        ul li a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Autoformalization Report</h1>
        
        <h2>Summary Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Problems</h3>
                <div class="value">{metrics['total_problems']}</div>
            </div>
            <div class="metric-card">
                <h3>Compilation Success</h3>
                <div class="value">{metrics['compilation_success_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Proof Success</h3>
                <div class="value">{metrics['proof_success_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Iterations</h3>
                <div class="value">{metrics['avg_iterations_to_success']:.2f}</div>
            </div>
        </div>
        
        <h2>Individual Reports</h2>
        <ul>
            {report_links}
        </ul>
    </div>
</body>
</html>
"""