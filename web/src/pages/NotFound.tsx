import { Compass } from "lucide-react";
import { Link } from "react-router-dom";
import { Card, Empty } from "../components/ui";

export default function NotFound() {
  return (
    <Card>
      <Empty icon={<Compass className="h-6 w-6" />} title="This page does not exist" action={<Link to="/" className="btn-primary btn-sm">Back to the dashboard</Link>}>
        Use the menu on the left to find every part of the system.
      </Empty>
    </Card>
  );
}
