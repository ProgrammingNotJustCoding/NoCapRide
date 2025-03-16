import Link from "next/link";
import { LuHouse } from "react-icons/lu";
import { GoGraph } from "react-icons/go";
import { PiTaxi } from "react-icons/pi";
import { FaRegUser } from "react-icons/fa";

const BottomNav: React.FC = () => {
  return (
    <nav className="w-full h-16 bg-gray-900 text-yellow-400 flex justify-around items-center shadow-lg">
      <Link href={"/"} className="flex flex-col items-center gap-1">
        <LuHouse className="text-2xl text-yellow-400" />
        <span className="text-xs">Home</span>
      </Link>
      <Link href={"/earnings"} className="flex flex-col items-center gap-1">
        <GoGraph className="text-2xl text-yellow-400" />
        <span className="text-xs">Earnings</span>
      </Link>
      <Link href={"/rides"} className="flex flex-col items-center gap-1">
        <PiTaxi className="text-2xl text-yellow-400" />
        <span className="text-xs">Rides</span>
      </Link>
      <Link href={"/account"} className="flex flex-col items-center gap-1">
        <FaRegUser className="text-2xl text-yellow-400" />
        <span className="text-xs">Account</span>
      </Link>
    </nav>
  );
};

export default BottomNav;