import streamlit as st
from PIL import Image

def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")
    
    st.sidebar.title("Tabs")
    tabs = ["Picture", "Parts Description", "About"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
    if selected_tab == "Picture":
        st.title("Picture")
        st.write("Here is the picture:")
        
        #picture_url = "https://example.com/picture.jpg"  # Replace with the URL of your picture
        picture_url = Image.open('drilling_rig.JPG')
        
        st.image(picture_url, use_column_width=True)
    
    elif selected_tab == "Parts Description":
        st.title("Parts Description")
        st.write("Click on the names to view the description of each part.")
        
        # Define the parts of the picture and their descriptions
        parts = {
            "Part A": "Description of Part A.",
            "Part B": "Description of Part B.",
            "Part C": "Description of Part C.",
        }
        
        # Display the arrows and names
        arrow_start_points = [(100, 200), (300, 400), (500, 600)]  # Replace with the coordinates of your arrows
        arrow_end_points = [(150, 250), (350, 450), (550, 650)]  # Replace with the coordinates of your arrows
        part_names = list(parts.keys())
        
        for i, start_point in enumerate(arrow_start_points):
            col1, col2, col3 = st.columns([0.4, 0.1, 0.5])
            
            with col1:
                st.write(part_names[i])
            
            with col2:
                st.write("->")
            
            with col3:
                # Add an empty space to reserve space for the description
                st.empty()
                
                # Get the description of the part when its name is clicked
                if col1.button(part_names[i]):
                    st.write(parts[part_names[i]])
    
    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")
    
if __name__ == "__main__":
    main()
