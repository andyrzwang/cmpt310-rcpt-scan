'''
From System Diagram.drawio
++ Process Steps: ++

Input receipt image

++ Step 1

    Image background removal
    Gray scale


++ Step 2
Text detection - CRAF
Fix image based on text detection output:

Upside down? or side ways:
    Rotation
    Align (tilt)
    expand scope

Use text detection again
- If converges, move to next step


'''