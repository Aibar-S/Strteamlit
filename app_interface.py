import streamlit as st
from PIL import Image

def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")
    
    st.sidebar.title("Tabs")
    tabs = ["Picture", "About"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
    if selected_tab == "Picture":
        st.title("Picture")
        
        # Define the parts of the picture and their descriptions
        parts = {
            "Part A": "Description of Part A.",
            "Part B": "Description of Part B.",
            "Part C": "Description of Part C.",
        }
        
        # Display the picture
        image = Image.open('drilling_rig.JPG')
        st.image(image, use_column_width=True)
        
        st.write("Click on the names to view the description of each part.")
        
        # Display the arrows and names on top of the picture
        arrow_start_points = [(100, 200), (300, 400), (500, 600)]  # Replace with the coordinates of your arrows
        arrow_end_points = [(150, 250), (350, 450), (550, 650)]  # Replace with the coordinates of your arrows
        part_names = list(parts.keys())
        
        # Create a container to hold the picture and the arrows
        container = st.container()
        
        # Render the picture
        container.image(picture_url, use_column_width=True)
        
        # Render the arrows and descriptions
        for i, start_point in enumerate(arrow_start_points):
            container.write(f"**{part_names[i]}**")
            container.image("https://example.com/arrow.png", use_column_width=True)
            
            # Get the description of the part when its name is clicked
            if container.button(part_names[i]):
                container.write(parts[part_names[i]])
    
    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")
    
if __name__ == "__main__":
    main()
