package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class Location {
    private final String locationId;
    private final String pageId;
    private final String pageTitle;
    private final String paragraphId;
    private final List<String> sectionId = new ArrayList<>();
    private final List<String> sectionHeadings = new ArrayList<>();

    public Location(@NotNull JSONObject jsonObject) throws JSONException {
        this.locationId = jsonObject.has("location_id")
                ?  jsonObject.getString("location_id")
                : " ";
        this.pageId = jsonObject.getString("page_id");
        this.pageTitle = jsonObject.getString("page_title");

        this.paragraphId = jsonObject.has("paragraph_id")
                    ? jsonObject.getString("paragraph_id")
                    : " ";

        JSONArray sectionIds = jsonObject.getJSONArray("section_id");
        for (int i = 0; i < sectionIds.length(); i++) {
            this.sectionId.add(sectionIds.getString(i));
        }

        JSONArray sectionHeadings = jsonObject.getJSONArray("section_headings");
        for (int i = 0; i < sectionHeadings.length(); i++) {
            this.sectionHeadings.add(sectionHeadings.getString(i));
        }
    }

    @Override
    public String toString() {
        return "Location{" +
                "locationId='" + locationId + '\'' +
                ", pageId='" + pageId + '\'' +
                ", pageTitle='" + pageTitle + '\'' +
                ", paragraphId='" + paragraphId + '\'' +
                ", sectionId=" + sectionId +
                ", sectionHeadings=" + sectionHeadings +
                '}';
    }

    public String getLocationId() {
        return locationId;
    }

    public String getPageId() {
        return pageId;
    }

    public String getPageTitle() {
        return pageTitle;
    }

    public String getParagraphId() {
        return paragraphId;
    }

    public List<String> getSectionId() {
        return sectionId;
    }

    public List<String> getSectionHeadings() {
        return sectionHeadings;
    }
}
