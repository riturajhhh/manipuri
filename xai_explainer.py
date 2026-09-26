import re
import numpy as np

def explain_text(text, predict_fn, clean_fn, classes, pred_class_name):
    """Generate perturbation-based word-level attributions for a predicted emotion class.
    
    Parameters:
    - text: original raw text input.
    - predict_fn: function taking raw text and returning full probability array (matching `classes` order).
    - clean_fn: text cleaning function.
    - classes: list of classes.
    - pred_class_name: string name of the predicted class.
    
    Returns a list of tuples: (token, score_percentage, is_word)
    """
    pred_class_lower = pred_class_name.lower()
    pred_idx = -1
    for idx, cls in enumerate(classes):
        if cls.lower() == pred_class_lower:
            pred_idx = idx
            break
            
    if pred_idx == -1:
        # Fallback if predicted class isn't found
        pred_idx = 0
        
    # Clean full text and get baseline probability
    cleaned_full = clean_fn(text)
    p_full = predict_fn(cleaned_full)
    prob_baseline = p_full[pred_idx]
    
    # Split text into tokens (words and spaces/punctuation)
    # We want to preserve spaces and punctuation so we can reconstruct the exact sentence in HTML
    tokens = re.split(r'(\s+|[.,!?\-:;()]+)', text)
    tokens = [t for t in tokens if t] # Filter out empty strings
    
    # Identify which tokens are actual words
    word_indices = []
    for idx, token in enumerate(tokens):
        # A token is a word if it contains letters, digits, or Indic characters
        if re.search(r'[\w\u0980-\u09FF\uABC0-\uABFF]', token):
            word_indices.append(idx)
            
    # Calculate word importance by perturbing (removing one word at a time)
    attributions = {}
    for w_idx in word_indices:
        # Create perturbed text omitting current word
        perturbed_tokens = tokens[:w_idx] + tokens[w_idx+1:]
        perturbed_text = "".join(perturbed_tokens)
        
        # Predict on perturbed text
        cleaned_perturbed = clean_fn(perturbed_text)
        p_perturbed = predict_fn(cleaned_perturbed)
        prob_perturbed = p_perturbed[pred_idx]
        
        # Attribution score: drop in probability when word is removed
        # Positive score = word contributed positively to the classification
        delta = prob_baseline - prob_perturbed
        attributions[w_idx] = delta
        
    # Normalize attributions
    # Sum of absolute attributions for scaling
    abs_sum = sum(abs(v) for v in attributions.values())
    if abs_sum == 0:
        abs_sum = 1e-5
        
    token_results = []
    for idx, token in enumerate(tokens):
        if idx in attributions:
            raw_score = attributions[idx]
            # Express as percent contribution
            pct = (raw_score / abs_sum) * 100
            token_results.append((token, pct, True))
        else:
            token_results.append((token, 0.0, False))
            
    return token_results

def generate_heatmap_html(token_results, emotion_color):
    """Convert token attribution results into a beautiful, styled HTML heatmap block.
    
    Supports hover tooltip styling and uses the predicted emotion's theme color for highlights.
    """
    html_out = []
    
    # Glassmorphism container for the heatmap
    html_out.append("""
    <div style="
        padding: 20px; 
        border-radius: 15px; 
        background: rgba(255, 255, 255, 0.02); 
        border: 1px solid rgba(255, 255, 255, 0.06); 
        line-height: 2.0; 
        font-size: 20px;
        text-align: center;
        margin-bottom: 25px;
        word-wrap: break-word;
    ">
    """)
    
    for token, score, is_word in token_results:
        if not is_word:
            # Render whitespace/punctuation plain
            html_out.append(f'<span style="color: #8b949e;">{token}</span>')
            continue
            
        # Select colors based on contribution sign
        if score > 0.5:
            # Positive contribution (strengthens emotion) - highlight in emotion theme color
            # Or standard green for positive, let's use the actual emotion color!
            # Convert hex to rgba
            hex_color = emotion_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Map score to opacity (0.1 to 0.75)
            opacity = 0.15 + (min(score, 80.0) / 80.0) * 0.6
            bg_style = f"background: rgba({r}, {g}, {b}, {opacity:.2f}); border-bottom: 2px solid rgba({r}, {g}, {b}, 0.8);"
            text_color = "#ffffff"
            tooltip_sign = "+"
        elif score < -0.5:
            # Negative contribution (weakens emotion / confuses) - highlight in red/orange
            opacity = 0.15 + (min(abs(score), 80.0) / 80.0) * 0.6
            bg_style = f"background: rgba(231, 76, 60, {opacity:.2f}); border-bottom: 2px solid rgba(231, 76, 60, 0.8);"
            text_color = "#ff9999"
            tooltip_sign = ""
        else:
            # Minimal contribution
            bg_style = "background: rgba(255, 255, 255, 0.04);"
            text_color = "#c9d1d9"
            tooltip_sign = "+"
            
        tooltip_text = f"Contribution: {tooltip_sign}{score:.1f}%"
        
        # Build token with tooltip and styles
        html_out.append(f"""
        <span class="xai-token" title="{tooltip_text}" style="
            display: inline-block;
            padding: 2px 8px;
            margin: 2px 4px;
            border-radius: 6px;
            {bg_style}
            color: {text_color};
            cursor: help;
            transition: all 0.2s ease;
            font-weight: 500;
        " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0px)'">
            {token}
        </span>
        """)
        
    html_out.append("</div>")
    return "".join(html_out)
